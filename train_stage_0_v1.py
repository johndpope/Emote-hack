import os
import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data  import DataLoader
from omegaconf import OmegaConf

from Net import FaceProjector,EMODataset

from typing import List, Dict, Any
# Other imports as necessary
import torch.optim as optim


def gpu_padded_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    assert isinstance(batch, list), "Batch should be a list"

    # Unpack and flatten the images, masks, downsampled activations, and mid-block activations from the batch
    all_images = []
    all_masks = []
    all_down_block_res_samples = [[] for _ in range(len(batch[0]['down_block_res_samples'][0]))]  # Create a list for each resolution
    all_mid_block_res_samples = []

    for item in batch:
        all_images.extend(item['images'])
        all_masks.extend(item['masks'])
        for i, down_block_res in enumerate(item['down_block_res_samples']):
            all_down_block_res_samples[i].extend(down_block_res)
        all_mid_block_res_samples.extend(item['mid_block_res_sample'])

    assert all(isinstance(img, torch.Tensor) for img in all_images), "All images must be PyTorch tensors"
    assert all(isinstance(mask, torch.Tensor) for mask in all_masks), "All masks must be PyTorch tensors"
    assert all(isinstance(res, torch.Tensor) for res_list in all_down_block_res_samples for res in res_list), "All downsampled activations must be PyTorch tensors"
    assert all(isinstance(res, torch.Tensor) for res in all_mid_block_res_samples), "All mid-block activations must be PyTorch tensors"

    # Determine the maximum dimensions for images and masks
    assert all(img.ndim == 3 for img in all_images), "All images must be 3D tensors"
    max_height = max(img.shape[1] for img in all_images)
    max_width = max(img.shape[2] for img in all_images)

    # Determine the maximum dimensions for downsampled activations
    max_heights_down = [max(res.shape[1] for res in res_list) for res_list in all_down_block_res_samples]
    max_widths_down = [max(res.shape[2] for res in res_list) for res_list in all_down_block_res_samples]

    # Determine the maximum dimensions for mid-block activations
    max_height_mid = max(res.shape[1] for res in all_mid_block_res_samples)
    max_width_mid = max(res.shape[2] for res in all_mid_block_res_samples)

    # Pad the images and masks
    padded_images = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in all_images]
    padded_masks = [F.pad(mask, (0, max_width - mask.shape[2], 0, max_height - mask.shape[1])) for mask in all_masks]

    # Pad the downsampled activations
    padded_down_block_res_samples = []
    for i, res_list in enumerate(all_down_block_res_samples):
        padded_res_list = [F.pad(res, (0, max_widths_down[i] - res.shape[2], 0, max_heights_down[i] - res.shape[1])) for res in res_list]
        padded_down_block_res_samples.append(torch.stack(padded_res_list))

    # Pad the mid-block activations
    padded_mid_block_res_samples = [F.pad(res, (0, max_width_mid - res.shape[2], 0, max_height_mid - res.shape[1])) for res in all_mid_block_res_samples]

    # Stack the padded images, masks, downsampled activations, and mid-block activations
    images_tensor = torch.stack(padded_images)
    masks_tensor = torch.stack(padded_masks)
    down_block_res_samples_tensor = tuple(padded_down_block_res_samples)
    mid_block_res_samples_tensor = torch.stack(padded_mid_block_res_samples)

    # Assert the correct shape of the output tensors
    assert images_tensor.ndim == 4, "Images tensor should be 4D"
    assert masks_tensor.ndim == 4, "Masks tensor should be 4D"
    assert all(res_tensor.ndim == 4 for res_tensor in down_block_res_samples_tensor), "All downsampled activation tensors should be 4D"
    assert mid_block_res_samples_tensor.ndim == 4, "Mid-block activation tensor should be 4D"

    return {'images': images_tensor, 'masks': masks_tensor, 'down_block_res_samples': down_block_res_samples_tensor, 'mid_block_res_sample': mid_block_res_samples_tensor}
def collate_fn(batch):
    # Define the maximum number of frames you want to consider per video
    max_frames_per_video = 100
    
    # Initialize lists to hold the processed images and masks
    batch_images = []
    batch_masks = []
    batch_video_ids = []
    
    # Process each item in the batch
    for item in batch:
        video_id = item['video_id']
        images = item['images']
        masks = item['masks']
        
        # Trim or pad the images and masks to have a uniform number of frames
        num_frames = len(images)
        
        if num_frames > max_frames_per_video:
            # Select the first 'max_frames_per_video' frames
            images = images[:max_frames_per_video]
            masks = masks[:max_frames_per_video]
        elif num_frames < max_frames_per_video:
            # Pad the sequences with zeros if they have less than 'max_frames_per_video' frames
            images.extend([torch.zeros_like(images[0])] * (max_frames_per_video - num_frames))
            masks.extend([torch.zeros_like(masks[0])] * (max_frames_per_video - num_frames))
        
        # Stack the images and masks along a new dimension
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        
        # Append the processed tensors to the batch lists
        batch_images.append(images)
        batch_masks.append(masks)
        batch_video_ids.append(video_id)
    
    # Combine the lists of tensors into single tensors
    batch_images = torch.stack(batch_images, dim=0)
    batch_masks = torch.stack(batch_masks, dim=0)
    
    # Return the batched data as a dictionary
    return {'video_id': batch_video_ids, 'images': batch_images, 'masks': batch_masks}


def train_model(model, data_loader, optimizer, criterion, device, num_epochs,cfg):
    model.train()  # Set the model to training mode
    
    # for param in model.parameters():
    #     print(param.name, param.requires_grad)
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch in data_loader:
            for i in range(batch['images'].size(0)):  # Iterate over images in the batch
                image = batch['images'][i].unsqueeze(0).to(device)  # Add batch dimension and move to device
                mask = batch['masks'][i].unsqueeze(0).to(device)  # Add batch dimension and move to device

                optimizer.zero_grad()  # Zero the parameter gradients
                output = model(image)  # Forward pass: compute the predicted mask
                loss = criterion(output, mask)  # Compute the loss
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
        # json_file='./data/celebvhq_info.json',
        stage='stage0-facelocator',
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
    model = FaceProjector().to(device)
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss when output is without sigmoid
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trained_model = train_model(model, data_loader, optimizer, criterion, device, num_epochs,cfg)

    # Save the model
    torch.save(trained_model.state_dict(), 'face_locator_model.pth')
    print("Model saved to face_locator_model.pth")


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage0.yaml")
    main(config)