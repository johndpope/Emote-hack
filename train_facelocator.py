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
from torch.nn.utils.rnn import pad_sequence



def padded_collate(batch):
    # Assuming each sample in the batch is a dictionary containing 'images' and 'masks' keys
    images = [sample['images'] for sample in batch]
    masks = [sample['masks'] for sample in batch]

    # Convert images and masks to tensors
    images = [torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in imgs]) for imgs in images]
    masks = [torch.stack([torch.from_numpy(np.array(mask)) for mask in mask_list]) for mask_list in masks]

    # Pad images and masks to the same size
    images = pad_sequence(images, batch_first=True)
    masks = pad_sequence(masks, batch_first=True)

    return images, masks


def train_model(model, data_loader, face_mask_generator, optimizer, criterion, device, num_epochs):
    assert isinstance(num_epochs, int) and num_epochs > 0, "Number of epochs must be a positive integer"
    model.train()

    for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                all_frame_images = batch["images"]

                for images in all_frame_images:
                    images = images.to(device)
                    assert images.ndim == 4, "Images must be a 4D tensor"
                    assert images.size(1) == 3, "Images must have 3 channels"

                    optimizer.zero_grad()

                    # Generate face masks for each image in the batch
                    face_masks = []
                    for img in images.cpu().numpy():
                        img = (img * 255).astype(np.uint8).transpose(1, 2, 0)
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