import torch
import torch.nn as nn
import torch.nn.functional as F

    

class Encoder(nn.Module):
    """
    Encoder encodes both reference and motion frames using a shared VAE encoder structure. 
    This simulates the behavior depicted in the Frames Encoding section of the diagram, 
    which indicates that reference and motion frames are passed through the same encoder.
    """
    def __init__(self, input_channels, latent_dim, img_size):
        super(Encoder, self).__init__()
        # Define the layers for the encoder according to the VAE structure.
        self.encoder_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * (img_size // 16) * (img_size // 16), latent_dim * 2),
        )
    
    def forward(self, x):
        # Pass input through encoder layers to get the latent space representation
        latent = self.encoder_layers(x)
        # Split the result into mu and logvar components of the latent space
        mu, logvar = torch.chunk(latent, 2, dim=-1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels,img_size):
        super(Decoder, self).__init__()
        self.img_size = img_size
        # The output size of the last deconvolution would be [output_channels, img_size, img_size]
        self.fc = nn.Linear(latent_dim, 256 * (img_size // 16) * (img_size // 16))
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 256, self.img_size // 16, self.img_size // 16)  # Reshape z to (batch_size, 256, img_size/16, img_size/16)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        reconstruction = torch.sigmoid(self.deconv4(z))  # Use sigmoid for normalizing the output to [0, 1]
        return reconstruction




class FramesEncodingVAE(nn.Module):
    """
    FramesEncodingVAE combines the encoding of reference and motion frames with additional components
    such as ReferenceNet and SpeedEncoder as depicted in the Frames Encoding part of the diagram.
    """
    def __init__(self, input_channels, latent_dim, img_size, reference_net):
        super(FramesEncodingVAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim, img_size)
        self.decoder = Decoder(latent_dim, input_channels, img_size)
        self.reference_net = reference_net

        # SpeedEncoder can be implemented as needed.
        self.speed_encoder = nn.Sequential(
            # Dummy layers for illustrative purposes; replace with actual speed encoding mechanism.
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, reference_image, motion_frames, speed_value):
        # Encode reference and motion frames
        reference_mu, reference_logvar = self.encoder(reference_image)
        motion_mu, motion_logvar = self.encoder(motion_frames)

        # Reparameterize reference and motion latent vectors
        reference_z = self.reparameterize(reference_mu, reference_logvar)
        motion_z = self.reparameterize(motion_mu, motion_logvar)

        # Process reference features with ReferenceNet
        reference_features = self.reference_net(reference_z)

        # Embed speed value
        speed_embedding = self.speed_encoder(speed_value)

        # Combine features
        combined_features = torch.cat([reference_features, motion_z, speed_embedding], dim=1)

        # Decode the combined features
        reconstructed_frames = self.decoder(combined_features)

        return reconstructed_frames, reference_mu, reference_logvar, motion_mu, motion_logvar

    def vae_loss(self, recon_frames, reference_image, motion_frames, reference_mu, reference_logvar, motion_mu, motion_logvar):
        # Reconstruction loss (MSE or BCE, depending on the final activation of the decoder)
        recon_loss = F.mse_loss(recon_frames, torch.cat([reference_image, motion_frames], dim=1), reduction='sum')
        
        # KL divergence loss for reference and motion latent vectors
        kl_loss_reference = -0.5 * torch.sum(1 + reference_logvar - reference_mu.pow(2) - reference_logvar.exp())
        kl_loss_motion = -0.5 * torch.sum(1 + motion_logvar - motion_mu.pow(2) - motion_logvar.exp())
        
        return recon_loss + kl_loss_reference + kl_loss_motion