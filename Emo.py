import os
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


reference_unet_config = {
    "sample_size": 256,                # The size of the input samples
    "in_channels": 3,                  # The number of input channels (e.g., for RGB images this is 3)
    "out_channels": 3,                 # The number of output channels
    "down_block_types": ("DownBlock2D",) * 4,   # A tuple defining the types of blocks in the downsampling path
    "up_block_types": ("UpBlock2D",) * 4,       # A tuple defining the types of blocks in the upsampling path
    # ... Additional configurations
}

denoising_unet_config = {
    "sample_size": 256,                # The size of the input samples
    "in_channels": 3,                  # The number of input channels (e.g., for RGB images this is 3)
    "out_channels": 3,                 # The number of output channels
    "down_block_types": ("DownBlock2D", "AttnDownBlock2D") * 2,   # A tuple defining the types of blocks, including attention blocks
    "up_block_types": ("UpBlock2D", "AttnUpBlock2D") * 2,         # A tuple defining the types of blocks, including attention blocks
    # ... Additional configurations
}

# Configuration for the EMOModel
config = {
    "num_speed_buckets": 10,
    "speed_embedding_dim": 64,
    "reference_unet_config": reference_unet_config,
    "denoising_unet_config": denoising_unet_config,
    # ... Additional model configurations
}


class EMOModel(nn.Module):
    def __init__(self, vae, image_encoder, config):
        super(EMOModel, self).__init__()
        self.vae = vae
        self.image_encoder = image_encoder

        # Reference UNet
        self.reference_unet = UNet2DConditionModel(**config.reference_unet_config)

        # Denoising UNet
        self.denoising_unet = UNet2DConditionModel(**config.denoising_unet_config)

        # Face Region Controller
        self.face_locator = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Speed Controller
        self.speed_embedding = nn.Embedding(config.num_speed_buckets, config.speed_embedding_dim)

    def forward(self, noisy_latents, timesteps, ref_image, motion_frames, audio_features, speed_embeddings):
        # Encode reference image
        ref_image_latents = self.vae.encode(ref_image).latent_dist.sample()
        ref_image_latents = ref_image_latents * 0.18215
        ref_image_embeds = self.image_encoder(ref_image)

        # Encode motion frames
        motion_frames_latents = self.vae.encode(motion_frames).latent_dist.sample()
        motion_frames_latents = motion_frames_latents * 0.18215
        motion_frames_embeds = self.image_encoder(motion_frames)

        # Get audio embeddings from the extracted features
        audio_embeds = audio_features

        # Compute face region mask
        face_region_mask = self.face_locator(ref_image)

        # Forward pass through Reference UNet
        self.reference_unet(ref_image_latents, encoder_hidden_states=ref_image_embeds)

        # Forward pass through Denoising UNet
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            audio_cond_fea=audio_embeds,
            pose_cond_fea=face_region_mask,
            speed_cond_fea=speed_embeddings,
            encoder_hidden_states=ref_image_embeds,
            motion_frames_hidden_states=motion_frames_embeds,
        )

        return model_pred
    
