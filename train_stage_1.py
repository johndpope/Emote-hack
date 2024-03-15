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



# works but complicated 
def gpu_padded_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

    assert isinstance(batch, list), "Batch should be a list"

    # Unpack and flatten the images and speeds from the batch
    all_images = []
    all_speeds = []
    for item in batch:
        # Assuming each 'images' field is a list of tensors for a single video
        all_images.extend(item['images'])  # Flatten the list of lists into a single list
        all_speeds.extend(item['speeds'])  # Flatten the list of lists into a single list


    assert all(isinstance(img, torch.Tensor) for img in all_images), "All images must be PyTorch tensors"
    assert all(isinstance(mask, torch.Tensor) for mask in all_speeds), "All speeds must be PyTorch tensors"


    # Determine the maximum dimensions
    assert all(img.ndim == 3 for img in all_images), "All images must be 3D tensors"
    max_height = max(img.shape[1] for img in all_images)
    max_width = max(img.shape[2] for img in all_images)

    # Pad the images and speeds
    padded_images = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in all_images]
    padded_speeds = [F.pad(mask, (0, max_width - mask.shape[2], 0, max_height - mask.shape[1])) for mask in all_speeds]


    # Stack the padded images and speeds
    images_tensor = torch.stack(padded_images)
    speeds_tensor = torch.stack(padded_speeds)

    # Assert the correct shape of the output tensors
    assert images_tensor.ndim == 4, "Images tensor should be 4D"
    assert speeds_tensor.ndim == 4, "speeds tensor should be 4D"
    
    return {'images': images_tensor, 'speeds': speeds_tensor}




def train_model(model, data_loader, optimizer, criterion, device, num_epochs, cfg):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in data_loader:
            for i in range(batch['images'].size(0)):  # Iterate over images in the batch
                # Ensure that we have enough previous frames for the current index
                start_idx = max(0, i - cfg.training.prev_frames)  # eg. 2 previous frames to consider
                end_idx = i + 1  # Exclusive end index for slicing

                reference_image = batch['images'][i].unsqueeze(0).to(device)
                prev_motion_frames = [batch['images'][j].unsqueeze(0).to(device) for j in range(start_idx, end_idx)]

                # Combine previous motion frames into a single tensor
                motion_frames = torch.cat(prev_motion_frames, dim=1).to(device)
                speed = batch['speeds'][i].unsqueeze(0).to(device)

                target_frames = torch.cat([reference_image] + prev_motion_frames, dim=1).to(device)

                optimizer.zero_grad()  # Zero the parameter gradients

                # Forward pass using the current reference image, previous motion frames, and speed
                recon_frames, _, _, _, _ = model(reference_image, motion_frames, speed)
                loss = criterion(recon_frames, target_frames)  # Compute the loss
                loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # Perform a single optimization step (parameter update)

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
        #json_file='./data/celebvhq_info.json',
        stage='stage1-vae',
        transform=transform
    )



    # Configuration and Hyperparameters
    num_epochs = 10  # Example number of epochs
    learning_rate = 1e-3  # Example learning rate

    # Initialize Dataset and DataLoader
    # Assuming EMODataset is properly defined and initialized as `dataset`
    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, collate_fn=gpu_padded_collate)

    # Model, Criterion, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FramesEncodingVAE(input_channels=3, latent_dim=256, img_size=cfg.data.train_height, reference_net=None).to(device)
    criterion = nn.MSELoss()  # Use MSE loss for VAE reconstruction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Train the model
    trained_model = train_model(model, data_loader, optimizer, criterion, device, num_epochs,cfg)


    # Save the model
    torch.save(trained_model.state_dict(), 'frames_encoding_vae_model.pth')
    print("Model saved to frames_encoding_vae_model.pth")


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1.yaml")
    main(config)