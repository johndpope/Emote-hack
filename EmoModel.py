
import torch.nn as nn
from diffusers import UNet2DConditionModel,UNet3DConditionModel
from diffusers.models.modeling_utils import ModelMixin




class EMOModel(ModelMixin):
    def __init__(self, vae, image_encoder, config):
        super().__init__()
        self.vae = vae
        self.image_encoder = image_encoder

        # Reference UNet
        self.reference_unet = UNet2DConditionModel(**config.reference_unet_config)

        # Denoising UNet
        self.denoising_unet = UNet3DConditionModel(
            sample_size=config.denoising_unet_config.get("sample_size"),
            in_channels=config.denoising_unet_config.get("in_channels"),
            out_channels=config.denoising_unet_config.get("out_channels"),
            down_block_types=config.denoising_unet_config.get("down_block_types"),
            up_block_types=config.denoising_unet_config.get("up_block_types"),
            block_out_channels=config.denoising_unet_config.get("block_out_channels"),
            layers_per_block=config.denoising_unet_config.get("layers_per_block"),
            cross_attention_dim=config.denoising_unet_config.get("cross_attention_dim"),
            attention_head_dim=config.denoising_unet_config.get("attention_head_dim"),
            use_motion_module=config.denoising_unet_config.get("use_motion_module"),
            motion_module_type=config.denoising_unet_config.get("motion_module_type"),
            motion_module_kwargs=config.denoising_unet_config.get("motion_module_kwargs"),
        )

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
        batch_size, num_frames, _, height, width = noisy_latents.shape

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
        face_region_mask = face_region_mask.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # Forward pass through Reference UNet
        ref_embeds = self.reference_unet(ref_image_latents, timesteps, ref_image_embeds).sample

        # Forward pass through Denoising UNet
        model_pred = self.denoising_unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=ref_embeds,
            pose_cond_fea=face_region_mask,
        ).sample

        return model_pred