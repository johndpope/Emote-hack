import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
from omegaconf import OmegaConf

class EMODatasetStage1(Dataset):
    """
    Stage 1 dataset focused purely on frame encoding.
    Only provides single frames for training the ReferenceNet and VAE.
    """
    def __init__(
        self,
        data_dir: str,
        video_dir: str,
        json_file: str,
        width: int = 512,
        height: int = 512,
        transform = None
    ):
        self.data_dir = Path(data_dir)
        self.video_dir = Path(video_dir)
        self.width = width
        self.height = height
        
        # Default transform if none provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load video metadata
        import json
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            self.video_ids = list(self.data['clips'].keys())

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single frame for training.
        For Stage 1, we only need individual frames.
        """
        video_id = self.video_ids[idx]
        video_path = self.video_dir / f"{video_id}.mp4"
        
        # Read a random frame from the video
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        
        # Get random frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = torch.randint(0, total_frames, (1,)).item()
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame from video: {video_path}")
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        
        # Apply transforms
        frame_tensor = self.transform(frame)
        
        return {
            'pixel_values': frame_tensor,
            'video_id': video_id
        }

class ReferenceNet(nn.Module):
    """
    ReferenceNet: Extracts reference features from input frames.
    Based on SD UNet architecture but modified for reference feature extraction.
    """
    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet
        
        # Freeze most UNet parameters except final blocks
        for name, param in self.unet.named_parameters():
            if 'up_blocks.3' not in name:  # Only fine-tune the final up block
                param.requires_grad = False

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Extract reference features through modified SD UNet."""
        return self.unet(latents, timesteps, return_dict=False)[0]

def train_stage1(config: OmegaConf) -> None:
    """
    Stage 1 training focusing on frame encoding with ReferenceNet and VAE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize dataset
    dataset = EMODatasetStage1(
        data_dir=config.data.data_dir,
        video_dir=config.data.video_dir,
        json_file=config.data.json_file,
        width=config.data.train_width,
        height=config.data.train_height
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    # Initialize models
    # 1. VAE from Stable Diffusion
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse"
    ).to(device)
    vae.eval()  # Freeze VAE weights
    
    # 2. UNet from Stable Diffusion for ReferenceNet
    reference_unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet"
    ).to(device)
    
    # 3. Initialize ReferenceNet
    reference_net = ReferenceNet(reference_unet).to(device)
    
    # Initialize optimizer (only for ReferenceNet)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, reference_net.parameters()),
        lr=config.training.learning_rate
    )
    
    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear"
    )
    
    # Training loop
    for epoch in range(config.training.num_epochs):
        total_loss = 0
        reference_net.train()
        
        for step, batch in enumerate(dataloader):
            # Get input images
            images = batch['pixel_values'].to(device)
            
            # Encode images to latent space using frozen VAE
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (images.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get model prediction
            noise_pred = reference_net(noisy_latents, timesteps)
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if step % config.training.log_every == 0:
                print(f"Epoch {epoch+1}/{config.training.num_epochs}, "
                      f"Step {step}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'reference_net_state_dict': reference_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(dataloader),
            }
            torch.save(
                checkpoint,
                f"{config.training.checkpoint_dir}/stage1_epoch_{epoch+1}.pt"
            )

if __name__ == "__main__":
    # Load config
    config = OmegaConf.load("configs/stage1.yaml")
    train_stage1(config)