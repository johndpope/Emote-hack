import os
import torch
import torch.nn as nn
#from EMOModel import EMOModel
from FaceMeshMaskGenerator import FaceMeshMaskGenerator
from FramesEncoder import FramesEncoder

from SpeedEncoder import SpeedEncoder
from VAEEncoder import VAE, ImageEncoder
from EMODataset import EMODataset
from Wav2VecFeatureExtractor import Wav2VecFeatureExtractor
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data  import DataLoader
import cv2
import mediapipe as mp
from diffusers.models.modeling_utils import ModelMixin
import argparse
from omegaconf import OmegaConf
import random
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm



def custom_collate(batch):
    collated_batch = {}

    for key in batch[0].keys():
        # Check if the elements under the current key are lists
        if isinstance(batch[0][key], list):
            # Convert each list to a tensor
            tensors = [torch.tensor(elem[key]) for elem in batch]

            # Assert that all tensors are of the same shape
            first_tensor_shape = tensors[0].shape
            assert all(tensor.shape == first_tensor_shape for tensor in tensors), f"Shape mismatch in batch for key {key}"

            # Stack the tensors
            collated_batch[key] = torch.stack(tensors)
        else:
            # Assert that all elements are tensors and have the same shape
            assert all(isinstance(elem[key], torch.Tensor) for elem in batch), f"Not all elements are tensors for key {key}"
            first_tensor_shape = batch[0][key].shape
            assert all(elem[key].shape == first_tensor_shape for elem in batch), f"Shape mismatch in batch for key {key}"

            # Stack the tensors
            collated_batch[key] = torch.stack([elem[key] for elem in batch])

    return collated_batch


# BACKBONE ~ MagicAnimate class
def main(cfg):

    
  

    # # Assuming you have the other required inputs (noisy_latents, timesteps, ref_image, speed_buckets)
    # # Instantiate the VAE and image encoder
    latent_dim = 256
    embedding_dim = 512
    # vae = VAE(latent_dim)
    # image_encoder = ImageEncoder(embedding_dim)

    # # Instantiate the EMOModel
    #emo_model = EMOModel(vae, image_encoder, config)


    # Create data loaders for each training stage
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    stage3_dataset = EMODataset(width=cfg.data.train_width,
            height=cfg.data.train_height,
            n_sample_frames=cfg.data.n_sample_frames,
            sample_rate=cfg.data.sample_rate,
            img_scale=(1.0, 1.0),
            data_dir='./images_folder', 
            video_dir='/home/oem/Downloads/CelebV-HQ/celebvhq/35666',
            json_file='./data/celebvhq_info.json', 
            stage='stage3', 
            transform=transform)
    stage3_dataloader = DataLoader(stage3_dataset, batch_size=16, shuffle=True, num_workers=0,collate_fn=custom_collate)

    num_speed_buckets = 9
    speed_embedding_dim = 64

    speed_layers = SpeedEncoder(num_speed_buckets, speed_embedding_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(speed_layers.parameters(), lr=0.001)


    num_epochs_stage3 = 2
    save_interval = 1  # Define how often you want to save the model (every 5 epochs in this case)

    # Stage 3: Speed Training
    for epoch in range(num_epochs_stage3):
        for batch in stage3_dataloader:
            if "all_head_rotation_speeds" in batch:
                # Loop through each video's frames in the batch
                for video_frames in batch["all_head_rotation_speeds"]:
                    for frame_speeds in video_frames:
                        # Check if the current frame has valid speed data
                        if frame_speeds.nelement() > 0:
                            # Process each frame individually
                            predicted_speeds = speed_layers(frame_speeds.unsqueeze(0))

                            # Compute loss
                            loss = criterion(predicted_speeds, frame_speeds.unsqueeze(0))

                            # Backpropagation and optimization
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # Optionally print out loss here for monitoring
                            print("loss:", loss.item())

        # Save the model at specified intervals
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs_stage3 - 1:
            save_path = os.path.join(cfg.model_save_dir, f'speed_encoder_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': speed_layers.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_path)
            print(f"Model saved at epoch {epoch + 1}")

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage3.yaml")
    main(config)
