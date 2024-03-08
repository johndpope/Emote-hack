import os
import torch
import torch.nn as nn
from FramesEncoder import FramesEncoder
from VAEEncoder import VAE, ImageEncoder
from Wav2VecFeatureExtractor import Wav2VecFeatureExtractor
from diffusers import UNet2DConditionModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model

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

    def forward(self, noisy_latents, timesteps, ref_image, motion_frames, audio_features):
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

        # Compute speed embedding
        speed_embeds = self.speed_embedding(speed_buckets)

        # Forward pass through Reference UNet
        self.reference_unet(ref_image_latents, encoder_hidden_states=ref_image_embeds)

        # Forward pass through Denoising UNet
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            audio_cond_fea=audio_embeds,
            pose_cond_fea=face_region_mask,
            speed_cond_fea=speed_embeds,
            encoder_hidden_states=ref_image_embeds,
            motion_frames_hidden_states=motion_frames_embeds,
        )

        return model_pred
    

feature_extractor = Wav2VecFeatureExtractor(model_name='facebook/wav2vec2-base-960h', device='cuda')
video_path = 'images_folder/M2Ohb0FAaJU_1'
audio_features = feature_extractor.extract_features_from_mp4(video_path, m=2, n=2)


frames_encoder = FramesEncoder()
reference_image_path = 'images_folder/M2Ohb0FAaJU_1/frame_0000.jpg'  
motion_frames_folder = 'images_folder/M2Ohb0FAaJU_1' 

reference_image_tensor = frames_encoder.encode_reference_image(reference_image_path)
motion_frames_tensor = frames_encoder.encode_motion_frames(motion_frames_folder)


# Assuming you have the other required inputs (noisy_latents, timesteps, ref_image, speed_buckets)
# Instantiate the VAE and image encoder
latent_dim = 256
embedding_dim = 512
vae = VAE(latent_dim)
image_encoder = ImageEncoder(embedding_dim)

# Instantiate the EMOModel
emo_model = EMOModel(vae, image_encoder, config)



def generate_noisy_latents(vae, timesteps, batch_size, latent_dim, device):
    # Sample latent vectors from the VAE
    latents = torch.randn(batch_size, latent_dim).to(device)
    latents = vae.decode(latents)

    # Add noise to the latents based on the timesteps
    noisy_latents = []
    for t in timesteps:
        noise = torch.randn_like(latents)
        noisy_latent = latents + noise * t
        noisy_latents.append(noisy_latent)

    noisy_latents = torch.stack(noisy_latents, dim=0)
    return noisy_latents

# Specify the necessary parameters
batch_size = 1
latent_dim = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_frames = len(os.listdir(motion_frames_folder))


timesteps = torch.linspace(0, 1, num_frames).to(device)  # Adjust the number of timesteps as needed

# Generate noisy latents
noisy_latents = generate_noisy_latents(vae, timesteps, batch_size, latent_dim, device)


output = emo_model(noisy_latents, timesteps, reference_image_tensor,motion_frames_tensor, audio_features)