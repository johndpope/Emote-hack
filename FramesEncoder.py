import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming you have a VAE and ReferenceNet defined elsewhere
# from vae_model import VariationalAutoEncoder
# from referencenet_model import ReferenceNet

class FramesEncoding(nn.Module):
    def __init__(self, vae_model, referencenet_model):
        super(FramesEncoding, self).__init__()
        self.vae = vae_model
        self.reference_net = referencenet_model

    def forward(self, reference_image, motion_frames):
        # Concatenate reference image and motion frames along the color channel axis
        # This assumes motion_frames is a tensor of shape (batch_size, num_frames, C, H, W)
        # and reference_image is a tensor of shape (batch_size, C, H, W)
        combined_input = torch.cat([reference_image.unsqueeze(1), motion_frames], dim=1)

        # Flatten the frames into the batch dimension to pass through the VAE
        batch_size, num_frames, C, H, W = combined_input.size()
        combined_input = combined_input.view(batch_size * num_frames, C, H, W)

        # Encode with VAE
        encoded_latents = self.vae.encode(combined_input).latent_dist.sample()

        # Reshape back to the original batch and frames dimension
        encoded_latents = encoded_latents.view(batch_size, num_frames, -1)

        # Pass the encoded latents through ReferenceNet
        reference_features = self.reference_net(encoded_latents)

        return reference_features