import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
import torchvision.transforms as transforms
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from PIL import Image
from pathlib import Path
import mediapipe as mp
import numpy as np
import cv2
import logging
import wandb
from omegaconf import OmegaConf
from typing import Dict, List, Tuple, Optional


class SpeedController(nn.Module):
    """
    Speed control module using bucketed embeddings as described in the paper.
    Controls head motion velocity across different video clips.
    """
    def __init__(self, num_buckets: int = 9, embed_dim: int = 1024):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed_dim = embed_dim
        
        # Initialize bucket centers and radii as per paper
        self.register_buffer('centers', torch.linspace(-1.0, 1.0, num_buckets))
        self.register_buffer('radii', torch.ones(num_buckets) * 0.1)
        
        # Speed embedding layers
        self.speed_embedding = nn.Embedding(num_buckets, embed_dim)
        self.speed_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def map_speed_to_bucket(self, speed: torch.Tensor) -> torch.Tensor:
        """Map continuous speed values to discrete buckets."""
        # Calculate distances to bucket centers
        distances = torch.abs(speed.unsqueeze(-1) - self.centers)
        # Return bucket indices of minimum distances
        return torch.argmin(distances, dim=-1)

    def forward(self, speeds: torch.Tensor) -> torch.Tensor:
        # Map speeds to buckets
        bucket_indices = self.map_speed_to_bucket(speeds)
        # Get embeddings
        embeddings = self.speed_embedding(bucket_indices)
        # Process through MLP
        return self.speed_mlp(embeddings)

class FaceRegionController(nn.Module):
    """
    Face region control module for maintaining consistent facial area generation.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1024):
        super().__init__()
        
        # Convolutional layers for processing face mask
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, 3, padding=1)
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(mask)

class EMODatasetStage3(Dataset):
    """
    Dataset for Stage 3 training, including speed and face region information.
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
        sample_rate: int = 16000
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.audio_ctx_frames = audio_ctx_frames
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Load metadata
        import json
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            self.video_ids = list(self.data['clips'].keys())
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _get_face_region_mask(self, frame: np.ndarray) -> torch.Tensor:
        """Generate face region mask using MediaPipe."""
        results = self.mp_face_mesh.process(frame)
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            points = np.array([(lm.x * self.width, lm.y * self.height) 
                             for lm in landmarks.landmark], dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 1.0)
        
        return torch.from_numpy(mask).unsqueeze(0)

    def _calculate_head_speed(self, 
                            curr_landmarks: List[mp.framework.formats.landmark_pb2.Landmark],
                            prev_landmarks: List[mp.framework.formats.landmark_pb2.Landmark]) -> float:
        """Calculate head rotation speed between frames."""
        def get_rotation_angles(landmarks):
            # Calculate head pose angles using specific landmark points
            # This is a simplified version - you might want to use more sophisticated methods
            nose = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
            left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
            right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])
            
            # Calculate rotation angles
            forward = nose - (left_eye + right_eye) / 2
            forward = forward / np.linalg.norm(forward)
            
            pitch = np.arcsin(forward[1])
            yaw = np.arctan2(forward[0], forward[2])
            
            return np.array([pitch, yaw])
        
        curr_angles = get_rotation_angles(curr_landmarks)
        prev_angles = get_rotation_angles(prev_landmarks)
        
        # Calculate angular velocity
        angle_diff = curr_angles - prev_angles
        speed = np.linalg.norm(angle_diff)
        
        # Normalize to [-1, 1] range
        return np.clip(speed / np.pi, -1.0, 1.0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[idx]
        video_path = str(self.video_dir / f"{video_id}.mp4")
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        face_masks = []
        landmarks_sequence = []
        
        # Read sequence of frames
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get face landmarks
            results = self.mp_face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                landmarks_sequence.append(results.multi_face_landmarks[0].landmark)
            
            # Get face mask
            face_mask = self._get_face_region_mask(frame_rgb)
            face_masks.append(face_mask)
            
            # Process frame
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = self.transform(frame_pil)
            frames.append(frame_tensor)
        
        cap.release()
        
        # Calculate speeds
        speeds = []
        for i in range(1, len(landmarks_sequence)):
            speed = self._calculate_head_speed(
                landmarks_sequence[i],
                landmarks_sequence[i-1]
            )
            speeds.append(speed)
        
        # Convert to tensors
        frames_tensor = torch.stack(frames)
        face_masks_tensor = torch.stack(face_masks)
        speeds_tensor = torch.tensor(speeds, dtype=torch.float32)
        
        return {
            'frames': frames_tensor,
            'face_masks': face_masks_tensor,
            'speeds': speeds_tensor,
            'video_id': video_id
        }

class EMOStage3(nn.Module):
    """
    Final stage model integrating all components including speed and face region control.
    """
    def __init__(
        self,
        temporal_unet: nn.Module,
        speed_controller: SpeedController,
        face_controller: FaceRegionController,
        vae: AutoencoderKL,
        audio_model: Wav2Vec2Model
    ):
        super().__init__()
        self.temporal_unet = temporal_unet
        self.speed_controller = speed_controller
        self.face_controller = face_controller
        self.vae = vae
        self.audio_model = audio_model

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        audio_features: torch.Tensor,
        speeds: torch.Tensor,
        face_masks: torch.Tensor,
        reference_frame: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get speed embeddings
        speed_embed = self.speed_controller(speeds)
        
        # Get face region features
        face_features = self.face_controller(face_masks)
        
        # Combine with input latents
        augmented_latents = noisy_latents + face_features
        
        # Pass through temporal UNet
        output = self.temporal_unet(
            augmented_latents,
            timesteps,
            audio_features,
            reference_frame
        )
        
        # Add speed control
        output = output + speed_embed.unsqueeze(-1).unsqueeze(-1)
        
        return output

class Stage3Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.training.device)
        self.logger = self._setup_logging()
        
        # Setup mixed precision training
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Initialize models and optimizer
        self.setup_models()
        self.setup_optimizer()
        
        # Initialize dataset and dataloader
        self.setup_data()
        
        # Setup wandb
        if config.training.use_wandb:
            self._setup_wandb()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.training.log_dir}/stage3.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _setup_wandb(self):
        wandb.init(
            project="emo-portrait",
            name=f"stage3-{wandb.util.generate_id()}",
            config=OmegaConf.to_container(self.config, resolve=True)
        )

    def setup_models(self):
        # Load pretrained Stage 2 model
        self.temporal_unet = torch.load(
            self.config.model.stage2_checkpoint,
            map_location=self.device
        )
        
        # Initialize new components
        self.speed_controller = SpeedController(
            num_buckets=self.config.model.num_speed_buckets,
            embed_dim=self.config.model.embed_dim
        ).to(self.device)
        
        self.face_controller = FaceRegionController(
            in_channels=1,
            out_channels=self.config.model.embed_dim
        ).to(self.device)
        
        # Load frozen models
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).to(self.device)
        self.vae.eval()
        
        self.audio_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        ).to(self.device)
        self.audio_model.eval()
        
        # Create full model
        self.model = EMOStage3(
            temporal_unet=self.temporal_unet,
            speed_controller=self.speed_controller,
            face_controller=self.face_controller,
            vae=self.vae,
            audio_model=self.audio_model
        ).to(self.device)

    def setup_optimizer(self):
        # Only optimize new components
        self.optimizer = torch.optim.AdamW([
            {'params': self.speed_controller.parameters()},
            {'params': self.face_controller.parameters()}
        ], lr=self.config.training.learning_rate)

    def setup_data(self):
        self.dataset = EMODatasetStage3(
            data_dir=self.config.data.data_dir,
            video_dir=self.config.data.video_dir,
            json_file=self.config.data.json_file,
            num_frames=self.config.data.num_frames,
            audio_ctx_frames=self.config.data.audio_ctx_frames
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )

    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        save_path = Path(self.config.training.checkpoint_dir) / f"stage3_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            frames = batch['frames'].to(self.device)
            face_masks = batch['face_masks'].to(self.device)
            speeds = batch['speeds'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.training.mixed_precision):
                # Encode frames to latent space
                with torch.no_grad():
                    latents = self.vae.encode(frames).latent_dist.sample()
                    latents = latents * 0.18215
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 1000, (frames.shape[0],),
                    device=self.device
                ).long()
                noisy_latents = latents + noise * timesteps.view(-1, 1, 1, 1, 1)
                
                # Process through model
                noise_pred = self.model(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    audio_features=None,  # Audio features from Stage 2
                    speeds=speeds,
                    face_masks=face_masks,
                    reference_frame=frames[:, 0]  # First frame as reference
                )
                
                # Calculate losses
                reconstruction_loss = F.mse_loss(noise_pred, noise)
                
                # Additional losses for face region and speed consistency
                face_region_loss = F.mse_loss(
                    noise_pred * face_masks,
                    noise * face_masks
                )
                
                # Total loss
                loss = reconstruction_loss + \
                       self.config.training.face_loss_weight * face_region_loss

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % self.config.training.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(self.dataloader)}, "
                    f"Loss: {avg_loss:.4f}"
                )
                
                if self.config.training.use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/reconstruction_loss": reconstruction_loss.item(),
                        "train/face_region_loss": face_region_loss.item(),
                        "train/step": epoch * len(self.dataloader) + batch_idx
                    })
        
        return total_loss / len(self.dataloader)

    def evaluate(self, epoch: int):
        """Evaluation step to monitor training progress."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                frames = batch['frames'].to(self.device)
                face_masks = batch['face_masks'].to(self.device)
                speeds = batch['speeds'].to(self.device)
                
                # Encode frames
                latents = self.vae.encode(frames).latent_dist.sample()
                latents = latents * 0.18215
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 1000, (frames.shape[0],),
                    device=self.device
                ).long()
                noisy_latents = latents + noise * timesteps.view(-1, 1, 1, 1, 1)
                
                # Model prediction
                noise_pred = self.model(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    audio_features=None,
                    speeds=speeds,
                    face_masks=face_masks,
                    reference_frame=frames[:, 0]
                )
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.dataloader)
        self.logger.info(f"Evaluation - Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        if self.config.training.use_wandb:
            wandb.log({
                "eval/loss": avg_loss,
                "eval/epoch": epoch
            })
        
        return avg_loss

    def train(self):
        """Main training loop for Stage 3."""
        self.logger.info("Starting Stage 3 training...")
        best_loss = float('inf')
        
        for epoch in range(self.config.training.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} training completed. Loss: {train_loss:.4f}")
            
            # Evaluation
            eval_loss = self.evaluate(epoch)
            
            # Save checkpoint
            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save_checkpoint(epoch, eval_loss)
                self.logger.info(f"New best model saved with loss: {eval_loss:.4f}")
            
            # Regular checkpoint saving
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(epoch, eval_loss)

def main():
    """Main function to run Stage 3 training."""
    # Load configuration
    config = OmegaConf.load("configs/stage3.yaml")
    
    # Create trainer
    trainer = Stage3Trainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()