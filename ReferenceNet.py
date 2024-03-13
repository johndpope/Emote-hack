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

class ReferenceNet(nn.Module):
    def __init__(self, vae_model, reference_net, speed_encoder):
        super(ReferenceNet, self).__init__()
        # Define the number of input channels and the scaling factor for feature channels
        num_channels = 3  # For RGB images
        feature_scale = 64  # Example scaling factor

        # Initialize the components
        self.vae = vae_model
        self.reference_net = reference_net
        self.speed_encoder = speed_encoder

        # Downsample and Upsample Blocks
        self.down1 = DownsampleBlock(num_channels, feature_scale)
        self.down2 = DownsampleBlock(feature_scale, feature_scale * 2)
        self.down3 = DownsampleBlock(feature_scale * 2, feature_scale * 4)
        self.up1 = UpsampleBlock(feature_scale * 4, feature_scale * 2)
        self.up2 = UpsampleBlock(feature_scale * 2, feature_scale)

        # Final convolution to adjust the number of output channels
        self.final_conv = nn.Conv2d(feature_scale, num_channels, kernel_size=1)

    def forward(self, reference_image, motion_frames, head_rotation_speed):
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

        # Final convolution to adjust the number of output channels
        out = self.final_conv(x)

        # Pass the output through ReferenceNet
        reference_features = self.reference_net(out)

        # Encode speed and expand its dimensions to concatenate with reference features
        speed_embedding = self.speed_encoder(head_rotation_speed)
        speed_embedding = speed_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, reference_features.size(2), reference_features.size(3))

        # Combine reference features and speed embedding
        combined_features = torch.cat([reference_features, speed_embedding], dim=1)

        return combined_features

