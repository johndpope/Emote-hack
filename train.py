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
    for key in batch[0]:
        elements = [d[key] for d in batch]
        # Check if the elements are tensors
        if all(isinstance(el, torch.Tensor) for el in elements):
            collated_batch[key] = torch.stack(elements)
        else:
            # Handle other data types (e.g., lists) appropriately
            # For instance, if elements are lists of different lengths, you might pad them
            # or handle them in a way that's appropriate for your application
            pass
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

    save_interval = 5  # Define how often you want to save the model (every 5 epochs in this case)


    # Stage 3: Speed Training
    for epoch in tqdm(range(num_epochs_stage3), desc='Epochs', unit='epoch'):
         # Initialize a progress bar for the batches in the current epoch
        batch_progress = tqdm(stage3_dataloader, desc=f'Epoch {epoch + 1}', leave=False, unit='batch')

        for batch in batch_progress:
            if len(batch.keys()) > 1:  # Just to check a couple of batches
        
                # print("batch:",batch)
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
        print('.')
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
