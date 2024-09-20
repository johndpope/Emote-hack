import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
import os
from typing import List, Tuple, Dict, Any
from decord import VideoReader, cpu
import torch.nn.functional as F


def gpu_padded_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    assert isinstance(batch, list), "Batch should be a list"
    
    # Determine the maximum number of frames across all videos in the batch
    max_frames = max(len(item['frames']) for item in batch)
    
    # Determine the maximum dimensions across all frames in the batch
    max_height = max(max(frame.shape[1] for frame in item['frames']) for item in batch)
    max_width = max(max(frame.shape[2] for frame in item['frames']) for item in batch)
    
    # Pad and stack frames for each item in the batch
    padded_frames = []
    for item in batch:
        frames = item['frames']
        # Pad each frame to max_height and max_width
        padded_item_frames = [F.pad(frame, (0, max_width - frame.shape[2], 0, max_height - frame.shape[1])) for frame in frames]
        # Pad the number of frames to max_frames
        while len(padded_item_frames) < max_frames:
            padded_item_frames.append(torch.zeros_like(padded_item_frames[0]))
        # Stack frames for this item
        padded_frames.append(torch.stack(padded_item_frames))
    
    # Stack all padded frame tensors in the batch
    frames_tensor = torch.stack(padded_frames)
    
    # Assert the correct shape of the output tensor
    assert frames_tensor.ndim == 5, "Frames tensor should be 5D (batch, frames, channels, height, width)"
    
    return {'frames': frames_tensor}

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_skip=0, num_frames=400):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.video_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        self.video_frames = self._count_frames()

    def _count_frames(self):
        video_frames = []
        for folder in self.video_folders:
            frames = [f for f in os.listdir(folder) if f.endswith('.png')]
            video_frames.append(len(frames))
        return video_frames

    def __len__(self):
        return len(self.video_folders)

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        transformed_images = [transform(img) for img in images]
        return torch.stack(transformed_images, dim=0)  # (f, c, h, w)

    def _load_and_transform_frame(self, frame_path):
        try:
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except (OSError, IOError) as e:
            print(f"Error loading image {frame_path}: {str(e)}")
            raise  # Re-raise the exception to be caught in __getitem__
        
    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        frames = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
        
        if not frames:
            print(f"No frames found in folder: {video_folder}")
            return self.__getitem__((idx + 1) % len(self))  # Move to next item
        
        total_frames = len(frames)
        if total_frames < self.num_frames:
            frame_indices = [i % total_frames for i in range(self.num_frames)]
        else:
            start_idx = random.randint(0, total_frames - self.num_frames)
            frame_indices = range(start_idx, start_idx + self.num_frames, self.frame_skip + 1)
        
        loaded_frames = []
        for i in frame_indices:
            img_path = os.path.join(video_folder, frames[i % total_frames])
            try:
                frame = self._load_and_transform_frame(img_path)
                loaded_frames.append(frame)
            except (OSError, IOError) as e:
                print(f"Error loading image {img_path}: {str(e)}")
                continue  # Skip this frame and continue with the next
        
        if not loaded_frames:
            print(f"No valid frames loaded from {video_folder}")
            return self.__getitem__((idx + 1) % len(self))  # Move to next item
        
        return {
            "frames": torch.stack(loaded_frames) if isinstance(loaded_frames[0], torch.Tensor) else loaded_frames,
            "video_name": os.path.basename(video_folder)
        }


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = VideoDataset("/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/images", 
                           transform=transform, 
                           frame_skip=0, 
                           num_frames=2)
    print(f"Total videos in dataset: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Number of PIL images: {len(sample['frames'])}")
    print(f"Tensor shape: {sample['tensor'].shape}")
    print(f"Video name: {sample['video_name']}")