import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
import yaml
import os
import torch.nn.functional as F

from VideoDataset import VideoDataset,gpu_padded_collate
from torchvision.utils import save_image
from helper import log_grad_flow,consistent_sub_sample,count_model_params,normalize,visualize_latent_token, add_gradient_hooks, sample_recon
from torch.optim import AdamW
from omegaconf import OmegaConf
import lpips
from torch.nn.utils import spectral_norm
import torchvision.models as models
import random


from torch.optim import AdamW, SGD
from transformers import Adafactor
from WebVid10M import WebVid10M
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers import UNet2DConditionModel
from diffusers import UNet2DModel

from diffusers.models.attention import CrossAttention


def load_config(config_path):
    return OmegaConf.load(config_path)

from Net import Wav2VecFeatureExtractor,FaceLocator,SpeedEncoder


class CustomUNet(UNet2DConditionModel):
    def __init__(self, config):
        super().__init__(**config)
        # Modify the cross-attention layers to include reference-attention
        for block in self.down_blocks + self.up_blocks:
            for layer in block.attentions:
                layer.transformer_blocks[0].attn1 = CrossAttention(
                    query_dim=layer.transformer_blocks[0].attn1.query_dim,
                    cross_attention_dim=layer.transformer_blocks[0].attn1.query_dim,
                    heads=layer.transformer_blocks[0].attn1.heads,
                    dim_head=layer.transformer_blocks[0].attn1.dim_head,
                    dropout=layer.transformer_blocks[0].attn1.dropout,
                )


class EMOModel(nn.Module):
    def __init__(self, config):
        super(EMOModel, self).__init__()
        self.config = config

        # Load the pretrained VAE (weights frozen)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae.eval()  # Freeze weights

        # Load the ReferenceNet (weights initialized from SD UNet)
        self.reference_unet =  UNet2DConditionModel.from_pretrained(
            '/media/2TB/Emote-hack/pretrained_models/StableDiffusion/stable-diffusion-v1-5',
            subfolder="unet",
        )
        self.reference_unet.requires_grad_(False)  # Freeze ReferenceNet during Stage 1

        # Load the Backbone Network (initialize weights from SD UNet)
        unet_config = self.reference_unet.config
        self.backbone_unet = CustomUNet(unet_config)

        # Initialize Backbone UNet weights from SD UNet
        self.backbone_unet.load_state_dict(self.reference_unet.state_dict())

        # Scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

    def forward(self, x_current, x_reference, timesteps):
        # Encode x_current and x_reference using VAE
        with torch.no_grad():
            z_current = self.vae.encode(x_current).latent_dist.sample() * 0.18215
            z_reference = self.vae.encode(x_reference).latent_dist.sample() * 0.18215

        # Add noise to z_current
        noise = torch.randn_like(z_current)
        z_noisy = self.scheduler.add_noise(z_current, noise, timesteps)

        # Get reference features from ReferenceNet
        with torch.no_grad():
            ref_outputs = self.reference_unet(
                sample=z_reference,
                timestep=timesteps,
                encoder_hidden_states=None,
                return_dict=True,
                output_hidden_states=True,
            )
            reference_hidden_states = ref_outputs.hidden_states  # Tuple of hidden states

        # Pass z_noisy and reference_hidden_states to the Backbone UNet
        backbone_outputs = self.backbone_unet(
            sample=z_noisy,
            timestep=timesteps,
            encoder_hidden_states=None,
            cross_attention_kwargs={'encoder_hidden_states': reference_hidden_states},
            return_dict=True,
        )

        noise_pred = backbone_outputs.sample

        return noise_pred, noise




class EmoTrainer:
    def __init__(self, config,   train_dataloader, accelerator):
        self.config = config
  
        self.train_dataloader = train_dataloader
        self.accelerator = accelerator


        # Initialize EMOModel
        self.model = EMOModel()


        # Initialize the scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

        # self.perceptual_loss_fn = lpips.LPIPS(net='alex', spatial=True).to(accelerator.device)
        self.pixel_loss_fn = nn.L1Loss()


        # Prepare models and optimizer
        self.optimizer = optim.AdamW(
            list(self.model.backbone_unet.parameters()) + list(self.model.reference_unet.parameters()),
            lr=config.training.learning_rate,
        )

        self.model,self.optimizer, self.scheduler, self.train_dataloader = accelerator.prepare(
           self.model,self.optimizer,self.scheduler,  self.train_dataloader
        )

class EmoTrainer:
    def __init__(self, config, train_dataloader, accelerator):
        self.config = config
        self.train_dataloader = train_dataloader
        self.accelerator = accelerator

        # Initialize EMOModel
        self.model = EMOModel(config)

        # Scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Prepare models and optimizer
        self.optimizer = optim.AdamW(
            self.model.backbone_unet.parameters(),
            lr=config.training.learning_rate,
        )

        self.model, self.optimizer, self.train_dataloader = accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

    def train_step(self, x_current, x_reference):
        # Sample random timesteps
        batch_size = x_current.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (batch_size,), device=self.accelerator.device
        ).long()

        # Forward pass through EMOModel
        noise_pred, noise = self.model(x_current, x_reference, timesteps)

        # Compute loss
        loss = self.loss_fn(noise_pred, noise)

        # Backpropagation
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()

        return loss.item()

    def train(self, start_epoch=0):
        global_step = start_epoch * len(self.train_dataloader)
        self.model.train()

        for epoch in range(self.config.training.num_epochs):
            progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}")

            for batch in self.train_dataloader:
                source_frames = batch['frames']
                batch_size, num_frames, channels, height, width = source_frames.shape

                # Randomly select reference and current frames
                ref_idx = torch.randint(0, num_frames, (1,)).item()
                x_reference = source_frames[:, ref_idx]

                current_idx = torch.randint(0, num_frames, (1,)).item()
                while current_idx == ref_idx:
                    current_idx = torch.randint(0, num_frames, (1,)).item()
                x_current = source_frames[:, current_idx]

                loss = self.train_step(x_current, x_reference)

                if self.accelerator.is_main_process and global_step % self.config.logging.log_every == 0:
                    wandb.log({"loss": loss, "global_step": global_step})

                global_step += 1

                # Optionally save checkpoints
                if global_step % self.config.training.save_steps == 0:
                    self.save_checkpoint(epoch, global_step)

                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{loss:.4f}"})

            progress_bar.close()

        # Final model saving
        self.save_checkpoint(epoch, global_step, is_final=True)


    def save_checkpoint(self, epoch, global_step, is_final=False):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        save_dir = self.config.training.output_dir
        os.makedirs(save_dir, exist_ok=True)
        filename = f'checkpoint_{global_step}.pth' if not is_final else 'final_checkpoint.pth'
        save_path = os.path.join(save_dir, filename)
        self.accelerator.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        self.accelerator.wait_for_everyone()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        print(f"Loaded checkpoint from {checkpoint_path} at epoch {epoch}, global step {global_step}")
        return epoch, global_step


def main():
    config = load_config('./configs/config.yaml')
    torch.cuda.empty_cache()
    wandb.init(project='EMO', config=OmegaConf.to_container(config, resolve=True))

    accelerator = Accelerator(
        mixed_precision=config.accelerator.mixed_precision,
        cpu=config.accelerator.cpu
    )



    # dataset = WebVid10M(video_folder=config.dataset.root_dir)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoDataset(config.dataset.root_dir, 
                                transform=transform, 
                                frame_skip=0, 
                                num_frames=300)

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=gpu_padded_collate 
    )


    trainer = EmoTrainer(config,   dataloader, accelerator)
    # Check if a checkpoint path is provided in the config
    if config.training.load_checkpoint:
        checkpoint_path = config.training.checkpoint_path
        start_epoch = trainer.load_checkpoint(checkpoint_path)
    else:
        start_epoch = 0
    trainer.train()

if __name__ == "__main__":
    main()