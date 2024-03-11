import math
import numpy as np
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin


# The Python code provided implements a SpeedEncoder as outlined in the whitepaper,
# with each bucket centering on specific head rotation velocities and radii. 
# It uses a hyperbolic tangent (tanh) function to scale the input speeds into a range between -1 and 1,
# creating a vector representing different velocity levels. 
# This vector is then processed through a multi-layer perceptron (MLP) to generate a speed embedding, 
# which can be utilized in downstream tasks such as controlling the speed and stability of generated animations. 
# This implementation allows for the synchronization of character's head motion across video clips, 
#     providing stable and controllable animation outputs.
class SpeedEncoder(ModelMixin):
    def __init__(self, num_speed_buckets, speed_embedding_dim):
        super().__init__()
        assert isinstance(num_speed_buckets, int), "num_speed_buckets must be an integer"
        assert num_speed_buckets > 0, "num_speed_buckets must be positive"
        assert isinstance(speed_embedding_dim, int), "speed_embedding_dim must be an integer"
        assert speed_embedding_dim > 0, "speed_embedding_dim must be positive"

        self.num_speed_buckets = num_speed_buckets
        self.speed_embedding_dim = speed_embedding_dim
        self.bucket_centers = self.get_bucket_centers()
        self.bucket_radii = self.get_bucket_radii()

        # Ensure that the length of bucket centers and radii matches the number of speed buckets
        assert len(self.bucket_centers) == self.num_speed_buckets, "bucket_centers length must match num_speed_buckets"
        assert len(self.bucket_radii) == self.num_speed_buckets, "bucket_radii length must match num_speed_buckets"

        self.mlp = nn.Sequential(
            nn.Linear(num_speed_buckets, speed_embedding_dim),
            nn.ReLU(),
            nn.Linear(speed_embedding_dim, speed_embedding_dim)
        )

    def get_bucket_centers(self):
        # Define the center values for each speed bucket
        # Adjust these values based on your specific requirements
        return [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]

    def get_bucket_radii(self):
        # Define the radius for each speed bucket
        # Adjust these values based on your specific requirements
        return [0.1] * self.num_speed_buckets

    def encode_speed(self, head_rotation_speed):
        # This method is now designed to handle a tensor of head rotation speeds
        # head_rotation_speed should be a 1D tensor of shape (batch_size,)
        assert head_rotation_speed.ndim == 1, "head_rotation_speed must be a 1D tensor"

        # Initialize a tensor to hold the encoded speed vectors
        speed_vectors = torch.zeros((head_rotation_speed.size(0), self.num_speed_buckets), dtype=torch.float32)

        for i in range(self.num_speed_buckets):
            center = self.bucket_centers[i]
            radius = self.bucket_radii[i]

            # Element-wise operation to compute the tanh encoding for each speed value in the batch
            speed_vectors[:, i] = torch.tanh((head_rotation_speed - center) / radius * 3)

        return speed_vectors

    def forward(self, head_rotation_speeds):
        # Ensure that head_rotation_speeds is a 1D Tensor of floats
        assert head_rotation_speeds.ndim == 1, "head_rotation_speeds must be a 1D tensor"
        assert head_rotation_speeds.dtype == torch.float32, "head_rotation_speeds must be a tensor of floats"

        # Process the batch of head rotation speeds through the encoder
        speed_vectors = self.encode_speed(head_rotation_speeds)

        # Pass the encoded vectors through the MLP
        speed_embeddings = self.mlp(speed_vectors)
        return speed_embeddings