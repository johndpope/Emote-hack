import math
import torch
import torch.nn as nn

class SpeedEncoder:
    def __init__(self, num_speed_buckets, speed_embedding_dim):
        self.num_speed_buckets = num_speed_buckets
        self.speed_embedding_dim = speed_embedding_dim
        self.bucket_centers = self.get_bucket_centers()
        self.bucket_radii = self.get_bucket_radii()
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
        speed_vector = np.zeros(self.num_speed_buckets)
        for i in range(self.num_speed_buckets):
            speed_vector[i] = math.tanh((head_rotation_speed - self.bucket_centers[i]) / self.bucket_radii[i] * 3)
        return speed_vector

    def forward(self, head_rotation_speeds):
        speed_vectors = [self.encode_speed(speed) for speed in head_rotation_speeds]
        speed_embeddings = self.mlp(torch.tensor(speed_vectors))
        return speed_embeddings