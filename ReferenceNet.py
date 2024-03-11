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
    def __init__(self, num_channels, feature_scale=64):
        super(ReferenceNet, self).__init__()
        self.down1 = DownsampleBlock(num_channels, feature_scale)
        self.down2 = DownsampleBlock(feature_scale, feature_scale * 2)
        self.down3 = DownsampleBlock(feature_scale * 2, feature_scale * 4)

        self.up1 = UpsampleBlock(feature_scale * 4, feature_scale * 2)
        self.up2 = UpsampleBlock(feature_scale * 2, feature_scale)
        self.final_conv = nn.Conv2d(feature_scale, num_channels, kernel_size=1)

    def forward(self, x):
        # Downsampling
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # Upsampling
        x = self.up1(x3, x2)
        x = self.up2(x, x1)

        # Final convolution
        out = self.final_conv(x)
        return out
