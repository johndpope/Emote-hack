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

def padded_collate(batch):
    assert isinstance(batch, list), "Batch should be a list"

    # Unpack the images and masks from the batch
    images = [item['images'] for item in batch]
    masks = [item['masks'] for item in batch]


    # Convert images and masks to tensors if they are not already
    images = [torch.tensor(img, dtype=torch.float32) if not isinstance(img, torch.Tensor) else img for img in images]
    masks = [torch.tensor(mask, dtype=torch.float32) if not isinstance(mask, torch.Tensor) else mask for mask in masks]

    # Assert that all images and masks are now tensors
    assert all(isinstance(img, torch.Tensor) for img in images), "All images must be PyTorch tensors"
    assert all(isinstance(mask, torch.Tensor) for mask in masks), "All masks must be PyTorch tensors"


    # Determine the maximum dimensions
    assert all(img.ndim == 3 for img in images), "All images must be 3D tensors"
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    # Pad the images and masks
    padded_images = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in images]
    padded_masks = [F.pad(mask, (0, max_width - mask.shape[2], 0, max_height - mask.shape[1])) for mask in masks]



    # Stack the padded images and masks
    images_tensor = torch.stack(padded_images)
    masks_tensor = torch.stack(padded_masks)

    # Assert the correct shape of the output tensors
    assert images_tensor.ndim == 4, "Images tensor should be 4D"
    assert masks_tensor.ndim == 4, "Masks tensor should be 4D"
    
    return {'images': images_tensor, 'masks': masks_tensor}


def train_model(model, data_loader, face_mask_generator, optimizer, criterion, device, num_epochs):
    assert isinstance(num_epochs, int) and num_epochs > 0, "Number of epochs must be a positive integer"
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
                all_frame_images = batch["images"]

                for images in all_frame_images:
                    images = images.to(device)
                    # [Rest of your image processing code]

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
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((cfg.data.train_height, cfg.data.train_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = EMODataset(
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

    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers,collate_fn=padded_collate)

    model = FaceLocator().to(device)
    face_mask_generator = FaceMaskGenerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = cfg.training.num_epochs

    trained_model = train_model(model, data_loader, face_mask_generator, optimizer, criterion, device, num_epochs)

    # Save the trained model
    save_path = os.path.join(cfg.model_save_dir, 'face_locator.pth')
    torch.save(trained_model.state_dict(), save_path)
    print(f"Trained model saved at: {save_path}")

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1.yaml")
    main(config)