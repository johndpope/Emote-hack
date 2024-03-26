import os
import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data  import DataLoader
from omegaconf import OmegaConf
from diffusers import  DDPMScheduler
from diffusers import UNet2DConditionModel
from magicanimate.models.unet_controlnet import UNet3DConditionModel
from Net import EMODataset,ReferenceNet
from typing import List, Dict, Any
from diffusers.models import AutoencoderKL
# Other imports as necessary
import torch.optim as optim
import yaml
from einops import rearrange
import torchvision.transforms as transforms



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# works but complicated 
def gpu_padded_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    assert isinstance(batch, list), "Batch should be a list"

    # Unpack and flatten the images, motion frames, and speeds from the batch
    all_images = []
  

    for item in batch:
        all_images.extend(item['images'])
     
 

    assert all(isinstance(img, torch.Tensor) for img in all_images), "All images must be PyTorch tensors"

    # Determine the maximum dimensions
    assert all(img.ndim == 3 for img in all_images), "All images must be 3D tensors"
    max_height = max(img.shape[1] for img in all_images)
    max_width = max(img.shape[2] for img in all_images)

    # Pad the images and motion frames
    padded_images = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in all_images]

    # Stack the padded images, motion frames, and speeds
    images_tensor = torch.stack(padded_images)


    # Assert the correct shape of the output tensors
    assert images_tensor.ndim == 4, "Images tensor should be 4D"

    return {'images': images_tensor}


# Q) should this spit out 64x64 or 32x32?
# The AutoencoderKL from the stabilityai/sd-vae-ft-mse configuration you've provided indicates that the sample_size is 256, which usually means the model is optimized for processing images of size 256x256 pixels

def images2latents(images, vae, dtype):
    """
    Encodes images to latent space using the provided VAE model.

    Args:
        images (torch.Tensor): Tensor of shape (batch_size, num_frames, channels, height, width) or (channels, height, width).
        vae (AutoencoderKL): Pre-trained VAE model.
        dtype (torch.dtype): Target data type for the latent tensors.

    Returns:
        torch.Tensor: Latent representations of the input images, reshaped as appropriate for conv2d input.
    """
    # Check if the input tensor has 3 or 4 dimensions and adjust accordingly
    # If the input is a single image (3D tensor), add a batch dimension
    if images.ndim == 3:
        images = images.unsqueeze(0)

    # Check if the input tensor has 4 or 5 dimensions and adjust accordingly
    if images.ndim == 5:
        # Combine batch and frames dimensions for processing
        images = images.view(-1, *images.shape[2:])
    
    # Resize the image to 256x256 before passing it to the VAE
    # resize_transform = transforms.Resize((256, 256))

    # Assuming 'images' is your 512x512 input tensor
    # resized_images = resize_transform(images)
    # Encode images to latent space and apply scaling factor
    latents = vae.encode(images.to(dtype=dtype)).latent_dist.sample()
    latents = latents * 0.18215

    # Ensure the output is 4D (batched input for conv2d)
    if latents.ndim == 5:
        latents = latents.view(-1, *latents.shape[2:])

    return latents.to(dtype=dtype)

def extract_features_from_model(model, data_loader, cfg):

        for batch in data_loader:
            video_frames = batch['images'].to(device)
            
            for i in range(1, video_frames.size(0)):

                if i < cfg.data.n_motion_frames:  # jump to the third frame - so we can get previous 2 frames
                    continue # diffused heads just uses the reference frame here instead. seems eroneous?

                reference_image = video_frames[i].unsqueeze(0)  #num_inference_frames Add batch dimension
                motion_frames = video_frames[max(0, i - cfg.data.n_motion_frames):i]  # add the 2 frames

                # Ensure the reference_image has the correct dimensions [1, C, H, W]
                assert reference_image.ndim == 4 and reference_image.size(0) == 1, "Reference image should have shape [1, C, H, W]"

                # Ensure motion_frames have the correct dimensions [N, C, H, W]
                assert motion_frames.ndim == 4, "Motion frames should have shape [N, C, H, W]"

   
                # Convert the reference image to latents
                reference_latent = images2latents(reference_image, dtype=model.dtype, vae=model.vae)
                print("reference_latent.ndim:",reference_latent.ndim)
                print("reference_latent.batch:",reference_latent.size(0))
                print("reference_latent.channels:",reference_latent.size(1))
                print("reference_latent.h:",reference_latent.size(2))
                print("reference_latent.w:",reference_latent.size(3))

                # 9 channels tensor? https://github.com/johndpope/Emote-hack/issues/27
                batch,latent_channels, height, width = reference_latent.shape

                # Convert the motion frames to latents and concatenate them with the reference latent IN THE CHANNEL DIMENSION
                motion_latents = []
                for idx, motion_frame in enumerate(motion_frames):
                    print("motion_frame.ndim:",motion_frame.ndim)
                    print("motion_frame.batch:",motion_frame.size(0))
                    print("motion_frame.channels:",motion_frame.size(1))
                    print("motion_frame.h:",motion_frame.size(2))
                    # print("motion_frame.w:",motion_frame.size(3))

                    motion_frame_latent = images2latents(motion_frame, dtype=model.dtype, vae=model.vae)
                    print("motion_frame_latent.ndim:",motion_frame_latent.ndim)
                    print("motion_frame_latent.b:",motion_frame_latent.size(0))
                    print("motion_frame_latent.c:",motion_frame_latent.size(1))
                    print("motion_frame_latent.h:",motion_frame_latent.size(2))
                    print("motion_frame_latent.w:",motion_frame_latent.size(3))

                    # Assert the shape of each motion frame latent
                    assert motion_frame_latent.shape == (batch,latent_channels, height, width), \
                        f"Motion frame latent {idx} has an inconsistent shape"
                    
                    motion_latents.append(motion_frame_latent)


                # Assuming reference_latent and motion_frame_latents are already computed
                motion_frame_latent1, motion_frame_latent2 = motion_latents  # Unpack the two motion frame latents

                # Concatenate the reference latent and one motion frame latent, then select channels to form a 9-channel tensor
                input_latent = torch.cat([
                    reference_latent[:, :3, :, :],  # Take first 3 channels
                    motion_frame_latent1[:, :3, :, :],  # Take first 3 channels from the first motion frame
                    motion_frame_latent2[:, :3, :, :]  # Take first 3 channels from the second motion frame
                ], dim=1)
                # Concatenate the reference latent and motion latents along the channel dimension
                # input_latent = torch.cat([reference_latent] + motion_latents, dim=0)

                print("input_latent.b:",input_latent.size(0))
                print("input_latent.c:",input_latent.size(1))
                print("input_latent.h:",input_latent.size(2))
                print("input_latent.w:",input_latent.size(3))
                


                # Pre-extract motion features from motion frames - white paper mentions this - but where?
                motion_features = model.pre_extract_motion_features(motion_frame_latent1)
                print("motion_features:",motion_features)
                
                motion_features = model.pre_extract_motion_features(motion_frame_latent2)
                print("motion_features:",motion_features)
                latent_features = model.pre_extract_motion_features(reference_latent)
                print("latent_features:",latent_features)
                





# Stage 1: Train the Referencenet
def main(cfg: OmegaConf) -> None:
    transform = transforms.Compose([
        transforms.Resize((cfg.data.train_height, cfg.data.train_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = EMODataset(
        use_gpu=cfg.training.use_gpu_video_tensor,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.data.n_sample_frames,
        sample_rate=cfg.data.sample_rate,
        img_scale=(1.0, 1.0),
        data_dir='./images_folder',
        video_dir=cfg.training.video_data_dir,
        json_file='./data/overfit.json',
        stage='stage1-0-framesencoder',
        transform=transform
    )



    # Initialize Dataset and DataLoader
    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, collate_fn=gpu_padded_collate)

    # Model, Criterion, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reference UNet (ReferenceNet)
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    reference_unet = UNet2DConditionModel.from_pretrained(
        '/media/2TB/Emote-hack/pretrained_models/StableDiffusion',
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=weight_dtype)

    referencenet = ReferenceNet(
        config=cfg,
        reference_unet=reference_unet,
        vae=vae,
        dtype=weight_dtype
    ).to(dtype=weight_dtype, device=device)


    # Extract Features
    extract_features_from_model(referencenet,data_loader,cfg)

    # Save the model
    # torch.save(trained_model.state_dict(), 'frames_encoding_vae_model.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1.yaml")
    main(config)