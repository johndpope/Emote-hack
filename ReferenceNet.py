import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim,img_size):
        super(Encoder, self).__init__()
        self.img_size = img_size
        # Simple convolutional architecture
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)  # Output size: [32, img_size/2, img_size/2]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # Output size: [64, img_size/4, img_size/4]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # Output size: [128, img_size/8, img_size/8]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # Output size: [256, img_size/16, img_size/16]
        self.fc_mu = nn.Linear(256 * (self.img_size // 16) * (self.img_size // 16), latent_dim)
        self.fc_logvar = nn.Linear(256 * (self.img_size // 16) * (self.img_size // 16), latent_dim)
       
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
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


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim,img_size):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim,img_size)
        self.decoder = Decoder(latent_dim, input_channels,img_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


    def vae_loss(recon_x, x, mu, logvar):
        # Reconstruction loss (MSE or BCE, depending on the final activation of the decoder)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    

class Net(nn.Module):
    def __init__(self, vae_model, reference_net):
        super(Net, self).__init__()
        # Define the number of input channels and the scaling factor for feature channels
        num_channels = 3  # For RGB images
        feature_scale = 64  # Example scaling factor
        self.vae = vae_model
        self.reference_net = reference_net

        self.down1 = DownsampleBlock(num_channels, feature_scale)
        self.down2 = DownsampleBlock(feature_scale, feature_scale * 2)
        self.down3 = DownsampleBlock(feature_scale * 2, feature_scale * 4)

        self.up1 = UpsampleBlock(feature_scale * 4, feature_scale * 2)
        self.up2 = UpsampleBlock(feature_scale * 2, feature_scale)
        self.final_conv = nn.Conv2d(feature_scale, num_channels, kernel_size=1)

    def forward(self, reference_image, motion_frames):
        # Downsample reference image
        ref_x1 = self.down1(reference_image)
        ref_x2 = self.down2(ref_x1)
        ref_x3 = self.down3(ref_x2)

        # Pass motion frames through similar downsampling blocks
        motion_x = motion_frames.view(-1, motion_frames.size(2), motion_frames.size(3), motion_frames.size(4))
        motion_x1 = self.down1(motion_x)
        motion_x2 = self.down2(motion_x1)
        motion_x3 = self.down3(motion_x2)
        
        # Upsample and integrate features from motion frames
        x = self.up1(ref_x3, motion_x3)
        x = self.up2(x, ref_x2)
        
        # Final convolution
        out = self.final_conv(x)
        
        # Pass the output through ReferenceNet
        reference_features = self.reference_net(out)
        
        return reference_features
