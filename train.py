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
    stage3_dataloader = DataLoader(stage3_dataset, batch_size=16, shuffle=True, num_workers=0)

    num_speed_buckets = 9
    speed_embedding_dim = 64

    speed_layers = SpeedEncoder(num_speed_buckets, speed_embedding_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(speed_layers.parameters(), lr=0.001)


    num_epochs_stage3 = 20

    save_interval = 5  # Define how often you want to save the model (every 5 epochs in this case)


    # Stage 3: Speed Training
    for epoch in range(num_epochs_stage3):
        for batch in stage3_dataloader:
            # Extract head rotation speeds from the batch
            head_rotation_speeds = batch["head_rotation_speeds"]  # Ground truth

            # Perform a forward pass through the SpeedEncoder
            # The SpeedEncoder's forward method now directly takes the head rotation speeds
            predicted_speeds = speed_layers(head_rotation_speeds)

            # Compute loss
            # The loss is calculated between the predicted speeds and the ground truth
            loss = criterion(predicted_speeds, head_rotation_speeds)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optionally print out loss here for monitoring
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Save the model at specified intervals
    if (epoch + 1) % save_interval == 0 or epoch == num_epochs_stage3 - 1:
        save_path = os.path.join(cfg.model_save_dir, f'speed_encoder_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': speed_layers.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),  # Save loss as a number
        }, save_path)
        print(f"Model saved at epoch {epoch + 1}")

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage3.yaml")
    main(config)
