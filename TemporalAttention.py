import torch
import torch.nn as nn
from magicanimate.models.motion_module import TemporalTransformer3DModel



class TemporalSelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=feature_dim,
            num_attention_heads=8,
            attention_head_dim=feature_dim // 8,
            num_layers=2,
            attention_block_types=("Temporal_Self", "Temporal_Self"),
            temporal_position_encoding=True,
            temporal_position_encoding_max_len=24,
        )

    def forward(self, x):
        # Reshape input feature map from (b × h × w) × f × c to b × c × f × h × w
        b, h, w, f, c = x.shape
        x = x.permute(0, 4, 3, 1, 2)

        # Apply temporal self-attention using the AnimateDiff models
        x = self.temporal_transformer(x)

        # Reshape the output back to (b × h × w) × f × c
        x = x.permute(0, 3, 4, 1, 2).reshape(b, h, w, f, c)
        return x

# Example usage
feature_dim = 512
num_frames = 10
batch_size = 4
height = 16
width = 16

# Create a sample input feature map
x = torch.randn(batch_size, height, width, num_frames, feature_dim)

# Create an instance of the TemporalSelfAttention module
temporal_attention = TemporalSelfAttention(feature_dim)

# Apply temporal self-attention to the input feature map
output = temporal_attention(x)

print(output.shape)  # Output shape: (batch_size, height, width, num_frames, feature_dim)