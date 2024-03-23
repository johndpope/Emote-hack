import os
import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data  import DataLoader
from omegaconf import OmegaConf

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
    all_motion_frames = []

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

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in data_loader:
            video_frames = batch['images'].to(device)
             # Process frames one by one, using previous frames as motion frames
            for i in range(1, video_frames.size(0)):
                reference_image = video_frames[i].unsqueeze(0)  # Add batch dimension
                motion_frames = video_frames[max(0, i-cfg.data.n_motion_frames):i].unsqueeze(0)  # Add batch dimension

                # Assert that reference_image and motion_frames have the expected shapes
                assert reference_image.size(0) == 1, "reference_image should have a batch size of 1"
                assert motion_frames.size(0) == 1, "motion_frames should have a batch size of 1"
                assert reference_image.size(1) == motion_frames.size(1), "reference_image and motion_frames should have the same number of channels"

                optimizer.zero_grad()

                # Forward pass
                recon_frames = model(reference_image, motion_frames)

                # Assert that recon_frames has the expected shape
                assert recon_frames.size(0) == 1, "recon_frames should have a batch size of 1"
                assert recon_frames.size(1) == reference_image.size(1) + motion_frames.size(1), "recon_frames should have the same number of channels as the concatenated reference_image and motion_frames"

                loss = criterion(recon_frames, torch.cat([reference_image, motion_frames], dim=1))
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


 

    model = FramesEncodingVAE(
        config=cfg
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