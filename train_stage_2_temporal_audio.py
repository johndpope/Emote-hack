import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
import torchvision.transforms as transforms
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import librosa
import numpy as np
from decord import VideoReader, AudioReader
import cv2

class EMODatasetStage2(Dataset):
    """
    Stage 2 dataset that provides consecutive frames and corresponding audio.
    """
    def __init__(
        self,
        data_dir: str,
        video_dir: str,
        json_file: str,
        num_frames: int = 8,
        audio_ctx_frames: int = 2,
        width: int = 512,
        height: int = 512,
        sample_rate: int = 16000,
        transform = None
    ):
        self.data_dir = Path(data_dir)
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.audio_ctx_frames = audio_ctx_frames
        self.sample_rate = sample_rate
        
        # Initialize wav2vec processor for audio features
        self.audio_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        
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

    def _load_audio_segment(self, video_path: str, start_frame: int, 
                          fps: float) -> torch.Tensor:
        """Load and process audio segment corresponding to frames."""
        # Extract audio using librosa
        y, sr = librosa.load(video_path, sr=self.sample_rate)
        
        # Calculate audio segment boundaries
        start_time = start_frame / fps
        end_time = (start_frame + self.num_frames) / fps
        
        # Get audio samples for the segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Get context window
        ctx_samples = int(self.audio_ctx_frames / fps * sr)
        start_sample = max(0, start_sample - ctx_samples)
        end_sample = min(len(y), end_sample + ctx_samples)
        
        audio_segment = y[start_sample:end_sample]
        
        # Process through wav2vec
        inputs = self.audio_processor(
            audio_segment, 
            sampling_rate=sr,
            return_tensors="pt"
        )
        
        return inputs.input_values[0]

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[idx]
        video_path = str(self.video_dir / f"{video_id}.mp4")
        
        # Read video
        vr = VideoReader(video_path)
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        # Get random sequence of consecutive frames
        start_idx = torch.randint(0, total_frames - self.num_frames, (1,)).item()
        frame_indices = range(start_idx, start_idx + self.num_frames)
        
        # Read and process frames
        frames = []
        for frame_idx in frame_indices:
            frame = vr[frame_idx].asnumpy()
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)
        
        # Stack frames
        frames = torch.stack(frames)
        
        # Get corresponding audio
        audio_features = self._load_audio_segment(video_path, start_idx, fps)
        
        return {
            'frames': frames,            # Shape: [T, C, H, W]
            'audio': audio_features,     # Shape: [A]
            'video_id': video_id
        }

class TemporalAttention(nn.Module):
    """Temporal self-attention module for frame sequence processing."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

class AudioAttention(nn.Module):
    """Cross-attention module for integrating audio features."""
    def __init__(self, frame_dim: int, audio_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (frame_dim // num_heads) ** -0.5
        
        self.frame_proj = nn.Linear(frame_dim, frame_dim)
        self.audio_proj = nn.Linear(audio_dim, frame_dim)
        self.out_proj = nn.Linear(frame_dim, frame_dim)

    def forward(self, frame_features: torch.Tensor, 
                audio_features: torch.Tensor) -> torch.Tensor:
        B, T, C = frame_features.shape
        
        # Project frame and audio features
        q = self.frame_proj(frame_features)
        k = v = self.audio_proj(audio_features)
        
        # Split heads
        q = q.reshape(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Merge heads
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.out_proj(x)
        return x

class TemporalUNet(nn.Module):
    """
    Enhanced UNet with temporal and audio attention modules.
    Built on top of Stage 1's ReferenceNet.
    """
    def __init__(
        self,
        reference_net: nn.Module,
        unet: UNet2DConditionModel,
        frame_dim: int = 1024,
        audio_dim: int = 768,
        num_frames: int = 8
    ):
        super().__init__()
        self.reference_net = reference_net
        self.unet = unet
        
        # Add temporal attention to each transformer block
        self.temporal_layers = nn.ModuleList([
            TemporalAttention(frame_dim) 
            for _ in range(len(self.unet.down_blocks))
        ])
        
        # Add audio attention after each temporal attention
        self.audio_layers = nn.ModuleList([
            AudioAttention(frame_dim, audio_dim)
            for _ in range(len(self.unet.down_blocks))
        ])
        
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        audio_features: torch.Tensor,
        reference_frame: torch.Tensor = None
    ) -> torch.Tensor:
        # Get reference features if provided
        ref_features = None
        if reference_frame is not None:
            ref_features = self.reference_net(reference_frame, timesteps)
        
        # Reshape latents to include temporal dimension
        B, T, C, H, W = latents.shape
        latents = latents.reshape(-1, C, H, W)
        
        # Initial convolution
        x = latents
        
        # Down blocks with temporal and audio attention
        down_block_res_samples = []
        for down_block, temporal_attn, audio_attn in zip(
            self.unet.down_blocks,
            self.temporal_layers,
            self.audio_layers
        ):
            x = down_block(x, timesteps, encoder_hidden_states=ref_features)
            
            # Apply temporal attention
            _, C, H, W = x.shape
            x = x.reshape(B, T, C, H, W)
            x_temp = temporal_attn(x.reshape(B, T, -1))
            x = x_temp.reshape(B, T, C, H, W)
            
            # Apply audio attention
            x_audio = audio_attn(x_temp, audio_features)
            x = x_audio.reshape(B, T, C, H, W)
            
            # Flatten temporal dimension for next block
            x = x.reshape(B * T, C, H, W)
            down_block_res_samples.append(x)
        
        # Middle block
        x = self.unet.mid_block(x, timesteps, encoder_hidden_states=ref_features)
        
        # Up blocks
        for up_block in self.unet.up_blocks:
            res_samples = down_block_res_samples[-1]
            down_block_res_samples = down_block_res_samples[:-1]
            x = up_block(x, res_samples, timesteps, 
                        encoder_hidden_states=ref_features)
        
        # Reshape output
        x = x.reshape(B, T, C, H, W)
        return x
class EMODatasetStage2(Dataset):
    """
    Stage 2 dataset that provides consecutive frames and corresponding audio.
    """
    def __init__(
        self,
        data_dir: str,
        video_dir: str,
        json_file: str,
        num_frames: int = 8,
        audio_ctx_frames: int = 2,
        width: int = 512,
        height: int = 512,
        sample_rate: int = 16000,
        transform = None
    ):
        self.data_dir = Path(data_dir)
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.audio_ctx_frames = audio_ctx_frames
        self.sample_rate = sample_rate
        
        # Initialize wav2vec processor for audio features
        self.audio_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        
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

    def _load_audio_segment(self, video_path: str, start_frame: int, 
                          fps: float) -> torch.Tensor:
        """Load and process audio segment corresponding to frames."""
        # Extract audio using librosa
        y, sr = librosa.load(video_path, sr=self.sample_rate)
        
        # Calculate audio segment boundaries
        start_time = start_frame / fps
        end_time = (start_frame + self.num_frames) / fps
        
        # Get audio samples for the segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Get context window
        ctx_samples = int(self.audio_ctx_frames / fps * sr)
        start_sample = max(0, start_sample - ctx_samples)
        end_sample = min(len(y), end_sample + ctx_samples)
        
        audio_segment = y[start_sample:end_sample]
        
        # Process through wav2vec
        inputs = self.audio_processor(
            audio_segment, 
            sampling_rate=sr,
            return_tensors="pt"
        )
        
        return inputs.input_values[0]

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[idx]
        video_path = str(self.video_dir / f"{video_id}.mp4")
        
        # Read video
        vr = VideoReader(video_path)
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        # Get random sequence of consecutive frames
        start_idx = torch.randint(0, total_frames - self.num_frames, (1,)).item()
        frame_indices = range(start_idx, start_idx + self.num_frames)
        
        # Read and process frames
        frames = []
        for frame_idx in frame_indices:
            frame = vr[frame_idx].asnumpy()
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)
        
        # Stack frames
        frames = torch.stack(frames)
        
        # Get corresponding audio
        audio_features = self._load_audio_segment(video_path, start_idx, fps)
        
        return {
            'frames': frames,            # Shape: [T, C, H, W]
            'audio': audio_features,     # Shape: [A]
            'video_id': video_id
        }

class TemporalAttention(nn.Module):
    """Temporal self-attention module for frame sequence processing."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

class AudioAttention(nn.Module):
    """Cross-attention module for integrating audio features."""
    def __init__(self, frame_dim: int, audio_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (frame_dim // num_heads) ** -0.5
        
        self.frame_proj = nn.Linear(frame_dim, frame_dim)
        self.audio_proj = nn.Linear(audio_dim, frame_dim)
        self.out_proj = nn.Linear(frame_dim, frame_dim)

    def forward(self, frame_features: torch.Tensor, 
                audio_features: torch.Tensor) -> torch.Tensor:
        B, T, C = frame_features.shape
        
        # Project frame and audio features
        q = self.frame_proj(frame_features)
        k = v = self.audio_proj(audio_features)
        
        # Split heads
        q = q.reshape(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Merge heads
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.out_proj(x)
        return x

class TemporalUNet(nn.Module):
    """
    Enhanced UNet with temporal and audio attention modules.
    Built on top of Stage 1's ReferenceNet.
    """
    def __init__(
        self,
        reference_net: nn.Module,
        unet: UNet2DConditionModel,
        frame_dim: int = 1024,
        audio_dim: int = 768,
        num_frames: int = 8
    ):
        super().__init__()
        self.reference_net = reference_net
        self.unet = unet
        
        # Add temporal attention to each transformer block
        self.temporal_layers = nn.ModuleList([
            TemporalAttention(frame_dim) 
            for _ in range(len(self.unet.down_blocks))
        ])
        
        # Add audio attention after each temporal attention
        self.audio_layers = nn.ModuleList([
            AudioAttention(frame_dim, audio_dim)
            for _ in range(len(self.unet.down_blocks))
        ])
        
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        audio_features: torch.Tensor,
        reference_frame: torch.Tensor = None
    ) -> torch.Tensor:
        # Get reference features if provided
        ref_features = None
        if reference_frame is not None:
            ref_features = self.reference_net(reference_frame, timesteps)
        
        # Reshape latents to include temporal dimension
        B, T, C, H, W = latents.shape
        latents = latents.reshape(-1, C, H, W)
        
        # Initial convolution
        x = latents
        
        # Down blocks with temporal and audio attention
        down_block_res_samples = []
        for down_block, temporal_attn, audio_attn in zip(
            self.unet.down_blocks,
            self.temporal_layers,
            self.audio_layers
        ):
            x = down_block(x, timesteps, encoder_hidden_states=ref_features)
            
            # Apply temporal attention
            _, C, H, W = x.shape
            x = x.reshape(B, T, C, H, W)
            x_temp = temporal_attn(x.reshape(B, T, -1))
            x = x_temp.reshape(B, T, C, H, W)
            
            # Apply audio attention
            x_audio = audio_attn(x_temp, audio_features)
            x = x_audio.reshape(B, T, C, H, W)
            
            # Flatten temporal dimension for next block
            x = x.reshape(B * T, C, H, W)
            down_block_res_samples.append(x)
        
        # Middle block
        x = self.unet.mid_block(x, timesteps, encoder_hidden_states=ref_features)
        
        # Up blocks
        for up_block in self.unet.up_blocks:
            res_samples = down_block_res_samples[-1]
            down_block_res_samples = down_block_res_samples[:-1]
            x = up_block(x, res_samples, timesteps, 
                        encoder_hidden_states=ref_features)
        
        # Reshape output
        x = x.reshape(B, T, C, H, W)
        return x
 

def train_stage2(config: OmegaConf) -> None:
    """Stage 2 training with temporal and audio integration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse"
    ).to(device)
    vae.eval()
    
    # Load pretrained audio model
    audio_model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h"
    ).to(device)
    audio_model.eval()
    
    # Load Stage 1 ReferenceNet
    reference_net = torch.load(
        config.stage1_checkpoint_path,
        map_location=device
    )
    
    # Initialize temporal UNet
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet"
    ).to(device)
    
    model = TemporalUNet(
        reference_net=reference_net,
        unet=unet,
        frame_dim=config.model.frame_dim,
        audio_dim=config.model.audio_dim,
        num_frames=config.data.num_frames
    ).to(device)
    
    # Initialize dataset and dataloader
    dataset = EMODatasetStage2(
        data_dir=config.data.data_dir,
        video_dir=config.data.video_dir,
        json_file=config.data.json_file,
        num_frames=config.data.num_frames,
        audio_ctx_frames=config.data.audio_ctx_frames
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            frames = batch['frames'].to(device)
            audio = batch['audio'].to(device)
            
            # Process audio through wav2vec
            with torch.no_grad():
                audio_features = audio_model(audio).last_hidden_state
            
            # Encode frames to latent space
            with torch.no_grad():
                latents = vae.encode(frames).latent_dist.sample()
                latents = latents * 0.18215
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (frames.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Use first frame as reference
            reference_frame = frames[:, 0]
            
            # Forward pass
            noise_pred = model(
                noisy_latents,
                timesteps,
                audio_features,
                reference_frame
            )
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if step % config.training.log_every == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(dataloader),
            }
            torch.save(
                checkpoint,
                f"{config.training.checkpoint_dir}/stage2_epoch_{epoch+1}.pt"
            )

if __name__ == "__main__":
    config = OmegaConf.load("configs/stage2.yaml")
    train_stage2(config)