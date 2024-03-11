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
        # Ensure that the head_rotation_speed is a scalar
        assert np.isscalar(head_rotation_speed), "head_rotation_speed must be a scalar value"
        # Check the value of head_rotation_speed is within the expected range, for example (-2, 2)
        assert -2 <= head_rotation_speed <= 2, "head_rotation_speed must be between -2 and 2"

        speed_vector = np.zeros(self.num_speed_buckets)
        for i in range(self.num_speed_buckets):
            speed_vector[i] = math.tanh((head_rotation_speed - self.bucket_centers[i]) / self.bucket_radii[i] * 3)
        return speed_vector

    def forward(self, head_rotation_speeds):
        # Assert that head_rotation_speeds is a 1D Tensor
        assert len(head_rotation_speeds.shape) == 1, "head_rotation_speeds must be a 1D tensor"
        # Assert that head_rotation_speeds is a tensor of floats
        assert head_rotation_speeds.dtype == torch.float, "head_rotation_speeds must be a tensor of floats"
        
        speed_vectors = [self.encode_speed(speed) for speed in head_rotation_speeds]
        speed_embeddings = self.mlp(torch.tensor(speed_vectors))
        return speed_embeddings