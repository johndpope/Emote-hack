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


def load_config(config_path):
    return OmegaConf.load(config_path)


class EMOModel(nn.Module):
    def __init__(self, vae, backbone_unet, reference_unet):
        super(EMOModel, self).__init__()
        self.vae = vae
        self.backbone_unet = backbone_unet
        self.reference_unet = reference_unet

    def forward(self, x_current, x_reference, timesteps, scheduler):
        # Encode frames using VAE
        with torch.no_grad():
            target_latent = self.vae.encode(x_current).latent_dist.sample() * 0.18215
            reference_latent = self.vae.encode(x_reference).latent_dist.sample() * 0.18215

        # Add noise to target latents
        noise = torch.randn_like(target_latent)
        noisy_target_latents = scheduler.add_noise(target_latent, noise, timesteps)

        # Extract reference features using ReferenceNet
        ref_outputs = self.reference_unet(reference_latent, timesteps)
        reference_features = ref_outputs.sample

        # Predict noise using Backbone Network
        noise_pred = self.backbone_unet(noisy_target_latents, timesteps, encoder_hidden_states=reference_features).sample

        return noise_pred, noise


class EmoTrainer:
    def __init__(self, config,   train_dataloader, accelerator):
        self.config = config
  
        self.train_dataloader = train_dataloader
        self.accelerator = accelerator

        # Load the pretrained VAE (weights frozen)
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.eval()  # Set to evaluation mode to freeze weights

        # Initialize Backbone Network (denoising UNet)
        backbone_unet = UNet2DConditionModel.from_pretrained(
            '/media/2TB/Emote-hack/pretrained_models/StableDiffusion/stable-diffusion-v1-5',
            subfolder="unet",
        )

        # Initialize ReferenceNet (similar to Backbone Network)
        reference_unet = UNet2DConditionModel.from_pretrained(
            '/media/2TB/Emote-hack/pretrained_models/StableDiffusion/stable-diffusion-v1-5',
            subfolder="unet",
        )

        # Initialize EMOModel
        self.model = EMOModel(vae=vae, backbone_unet=backbone_unet, reference_unet=reference_unet)


        # Initialize the scheduler
        scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.perceptual_loss_fn = lpips.LPIPS(net='alex', spatial=True).to(accelerator.device)
        self.pixel_loss_fn = nn.L1Loss()


        # Prepare models and optimizer
        self.optimizer = optim.AdamW(
            list(self.backbone_unet.parameters()) + list(self.reference_unet.parameters()),
            lr=config.training.learning_rate,
        )

        self.model,self.optimizer,  self.train_dataloader = accelerator.prepare(
           self.model,self.optimizer,  self.train_dataloader
        )


    def train_step(self, x_current, x_reference, global_step):
        if x_current.nelement() == 0:
            print("🔥 Skipping training step due to empty x_current")
            return None, None, None, None, None, None



        # Sample random timesteps
        batch_size = x_current.shape[0]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), device=self.device).long()


        #Forward pass through EMOModel
        noise_pred, noise = self.model(x_current, x_reference, timesteps, self.scheduler)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # Backpropagation
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()

        return loss.item()

    def train(self, start_epoch=0):
        global_step = start_epoch * len(self.train_dataloader)


        for epoch in range(self.config.training.num_epochs):
             

            self.model.train()
            progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}")

    
            num_valid_steps = 0
 
            for batch in self.train_dataloader:
                source_frames = batch['frames']
                batch_size, num_frames, channels, height, width = source_frames.shape

                for _ in range(int(1)):
                    if self.config.training.use_many_xrefs:
                        ref_indices = range(0, num_frames, self.config.training.every_xref_frames)
                    else:
                        ref_indices = [0]

                    for ref_idx in ref_indices:
                        x_reference = source_frames[:, ref_idx]

                        for current_idx in range(num_frames):
                            if current_idx == ref_idx:
                                continue

                            x_current = source_frames[:, current_idx]

                            results = self.train_step(x_current, x_reference, global_step)

                            if results[0] is not None:
                                noise_loss  = results
                                epoch_noise_loss += noise_loss
                                num_valid_steps += 1

                            else:
                                print("Skipping step due to error in train_step")

                            if self.accelerator.is_main_process and global_step % self.config.logging.log_every == 0:
                                wandb.log({
                                    "epoch_noise_loss": epoch_noise_loss,
                                    "global_step": global_step
                                
                                })
                                # Log gradient flow for generator and discriminator
                                # log_grad_flow(self.model.named_parameters(),global_step)


                            # if global_step % self.config.logging.sample_every == 0:
                            #     sample_path = f"recon_step_{global_step}.png"
                            #     sample_recon(self.model, (x_reconstructed, x_current, x_reference), self.accelerator, sample_path, 
                            #                  num_samples=self.config.logging.sample_size)
                                
                            global_step += 1

                             # Checkpoint saving
                            if global_step % self.config.training.save_steps == 0:
                                self.save_checkpoint(epoch)

                # Calculate average losses for the epoch
                if num_valid_steps > 0:
                    avg_g_loss = epoch_noise_loss / num_valid_steps


                progress_bar.update(1)
                progress_bar.set_postfix({"Noise Loss": f"{epoch_noise_loss:.4f}"})

            progress_bar.close()
            

        # Final model saving
        self.save_checkpoint(epoch, is_final=True)


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