import os
import random
import numpy as np
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class WebVid10M(Dataset):
    def __init__(
            self,
            video_folder,
            sample_size=256,
            num_frames=200  # Set a fixed number of frames per video
        ):
        self.dataset = [os.path.join(video_folder, video_path) for video_path in os.listdir(video_folder) if video_path.endswith(("mp4",))]
        random.shuffle(self.dataset)
        self.length = len(self.dataset)

        self.video_folder = video_folder
        self.num_frames = num_frames
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        ])
    
    def get_batch(self, idx):
        video_dir = self.dataset[idx]
        name = os.path.basename(video_dir)
        
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        # Calculate how many times we need to repeat the video
        repeat_times = self.num_frames // video_length + 1
        
        # Create an index list that covers all frames and repeats if necessary
        batch_index = list(range(video_length)) * repeat_times
        batch_index = batch_index[:self.num_frames]
        
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break
            except Exception as e:
                print(f"Error loading video {self.dataset[idx]}: {e}")
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
     
        return {
            "frames": pixel_values,
            "video_name": name
        }