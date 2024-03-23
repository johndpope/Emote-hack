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
from Net import FaceLocator,EMODataset,FramesEncodingVAE
from typing import List, Dict, Any
# Other imports as necessary
import torch.optim as optim
import yaml


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


def train_model(model, data_loader, optimizer, criterion, device, num_epochs, cfg):
    model.train()

    # Create the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.noise_scheduler_kwargs.num_train_timesteps,
        beta_start=cfg.noise_scheduler_kwargs.beta_start,
        beta_end=cfg.noise_scheduler_kwargs.beta_end,
        beta_schedule=cfg.noise_scheduler_kwargs.beta_schedule,
        steps_offset=cfg.noise_scheduler_kwargs.steps_offset,
        clip_sample=cfg.noise_scheduler_kwargs.clip_sample,
    )

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in data_loader:
            video_frames = batch['images'].to(device)
            
            for i in range(1, video_frames.size(0)):
                if i < cfg.data.n_motion_frames: # jump to the third frame - so we can get previous 2 frames
                    continue

                reference_image = video_frames[i].unsqueeze(0)  # Add batch dimension
                motion_frames = video_frames[max(0, i - cfg.data.n_motion_frames):i] # add the 2 frames

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (reference_image.shape[0],), device=reference_image.device)
                
                # Add noise to the reference image
                noisy_reference_image = noise_scheduler.add_noise(reference_image, torch.randn_like(reference_image), timesteps)

                optimizer.zero_grad()

                # Prepare input for the model
                encoder_hidden_states = model.reference_unet.encode_hidden_state(reference_image, timesteps)

                # Forward pass, ensure all required arguments are passed
                recon_frames = model(noisy_reference_image, motion_frames, timestep=timesteps, encoder_hidden_states=encoder_hidden_states)

                # Calculate loss
                loss = criterion(recon_frames, reference_image)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return model


# BACKBONE ~ MagicAnimate class
# Stage 1: Train the VAE (FramesEncodingVAE) with the Backbone Network and FaceLocator.
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
        video_dir='/home/oem/Downloads/CelebV-HQ/celebvhq/35666',
        json_file='./data/overfit.json',
        stage='stage1-0-framesencoder',
        transform=transform
    )

    # Configuration and Hyperparameters
    num_epochs = 10  # Example number of epochs
    learning_rate = 1e-3  # Example learning rate

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
    
    inference_config = OmegaConf.load("configs/inference.yaml")
        
    denoising_unet = UNet3DConditionModel.from_pretrained_2d('/media/2TB/ani/animate-anyone/pretrained_models/sd-image-variations-diffusers', subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()
       

    # denoising_unet = UNet3DConditionModel.from_pretrained(
    #     '/media/2TB/Emote-hack/pretrained_models/StableDiffusion', 
    #     subfolder='unet', torch_dtype=weight_dtype, device=device)


    model = FramesEncodingVAE(
        config=cfg,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet
    ).to(device)
    criterion = nn.MSELoss()  # Use MSE loss for VAE reconstruction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trained_model = train_model(model, data_loader, optimizer, criterion, device, num_epochs, cfg)

    # Save the model
    torch.save(trained_model.state_dict(), 'frames_encoding_vae_model.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1.yaml")
    main(config)