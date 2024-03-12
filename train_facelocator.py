import os
import torch
import torch.nn as nn

import torch.nn.functional as F

from EMODataset import EMODataset
import torchvision.transforms as transforms
from torch.utils.data  import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from FaceLocator import FaceLocator,FaceMaskGenerator
import numpy as np
from torchvision.transforms.functional import pad
from typing import List, Dict, Any



# works but complicated 
def gpu_padded_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

    assert isinstance(batch, list), "Batch should be a list"

    # Unpack and flatten the images and masks from the batch
    all_images = []
    all_masks = []
    for item in batch:
        # Assuming each 'images' field is a list of tensors for a single video
        all_images.extend(item['images'])  # Flatten the list of lists into a single list
        all_masks.extend(item['masks'])  # Flatten the list of lists into a single list


    assert all(isinstance(img, torch.Tensor) for img in all_images), "All images must be PyTorch tensors"
    assert all(isinstance(mask, torch.Tensor) for mask in all_masks), "All masks must be PyTorch tensors"


    # Determine the maximum dimensions
    assert all(img.ndim == 3 for img in all_images), "All images must be 3D tensors"
    max_height = max(img.shape[1] for img in all_images)
    max_width = max(img.shape[2] for img in all_images)

    # Pad the images and masks
    padded_images = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in all_images]
    padded_masks = [F.pad(mask, (0, max_width - mask.shape[2], 0, max_height - mask.shape[1])) for mask in all_masks]


    # Stack the padded images and masks
    images_tensor = torch.stack(padded_images)
    masks_tensor = torch.stack(padded_masks)

    # Assert the correct shape of the output tensors
    assert images_tensor.ndim == 4, "Images tensor should be 4D"
    assert masks_tensor.ndim == 4, "Masks tensor should be 4D"
    
    return {'images': images_tensor, 'masks': masks_tensor}




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

def train_model(model: nn.Module, data_loader: DataLoader, face_mask_generator: FaceMaskGenerator,
                optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, num_epochs: int,cfg:OmegaConf) -> nn.Module:
   

    model.train()

    for epoch in range(0, num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(data_loader):
                print("batch:",batch)

                all_frame_images = batch["images"]

                for images in all_frame_images:

                    images = images.to(device)
                 

                    # Generate face masks for each image in the batch
                    face_masks = []
                    for img in images.cpu():
                       # Convert the tensor to a numpy array and transpose it to HWC format
                        img = img.numpy().transpose(1, 2, 0)
                        # Ensure the image is in uint8 format
                        img = (img * 255).astype(np.uint8)

                        mask = face_mask_generator.generate_mask(img)
                        face_masks.append(mask)

                    # Convert list of masks to a tensor
                    face_masks_tensor = torch.stack(face_masks).to(device)

                    # Forward pass: compute the output of the model using images and masks
                    outputs = model(images, face_masks_tensor)

                    # Ensure the mask is the same shape as the model's output
                    face_masks_tensor = F.interpolate(face_masks_tensor.unsqueeze(1), size=outputs.shape[2:], mode='nearest').squeeze(1)
                    assert outputs.shape == face_masks_tensor.shape, "Output and face masks tensor must have the same shape"

                    # Compute the loss
                    loss = criterion(outputs, face_masks_tensor)
                    epoch_loss += loss.item()

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    epoch_loss /= len(data_loader)
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return model




# BACKBONE ~ MagicAnimate class
def main(cfg: OmegaConf) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        json_file='./data/celebvhq_info.json',
        stage='stage1',
        transform=transform
    )

    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers,collate_fn=gpu_padded_collate)

    model = FaceLocator().to(device)
    face_mask_generator = FaceMaskGenerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = cfg.training.num_epochs

    trained_model = train_model(model, data_loader, face_mask_generator, optimizer, criterion, device, num_epochs,cfg)

    # Save the trained model
    save_path = os.path.join(cfg.model_save_dir, 'face_locator.pth')
    torch.save(trained_model.state_dict(), save_path)
    print(f"Trained model saved at: {save_path}")

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1.yaml")
    main(config)