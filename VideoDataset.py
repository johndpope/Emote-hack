import os
import json
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from Wav2VecFeatureExtractor import Wav2VecFeatureExtractor
from HeadRotation import get_head_pose_velocities

class EMODataset(Dataset):
    def __init__(self, data_dir, audio_dir, json_file, transform=None):
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.transform = transform
        self.feature_extractor = Wav2VecFeatureExtractor(model_name='facebook/wav2vec2-base-960h', device='cuda')

        with open(json_file, 'r') as f:
            self.celebvhq_info = json.load(f)

        self.video_ids = list(self.celebvhq_info['clips'].keys())

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video_info = self.celebvhq_info['clips'][video_id]
        ytb_id = video_info['ytb_id']

        frame_folder = os.path.join(self.data_dir, f"{ytb_id}_{video_id.split('_')[-1]}")
        # audio_path = os.path.join(self.audio_dir, f"{ytb_id}_{video_id.split('_')[-1]}.wav")
        mp4_path = os.path.join(self.audio_dir, f"{ytb_id}_{video_id.split('_')[-1]}.mp4")
        frames = sorted([frame for frame in os.listdir(frame_folder) if frame.endswith(".jpg")])
        reference_frame = frames[0]  # Use the first frame as the reference frame
        motion_frames = frames[1:]  # Use the remaining frames as motion frames

        reference_image = Image.open(os.path.join(frame_folder, reference_frame))
        motion_frame_paths = [os.path.join(frame_folder, frame) for frame in motion_frames]
        
        if self.transform:
            reference_image = self.transform(reference_image)

        audio_features = self.feature_extractor.extract_features_from_mp4(mp4_path, m=2, n=2)
        
        head_rotation_speeds = get_head_pose_velocities(motion_frame_paths)

        sample = {
            "video_id": video_id,
            "reference_image": reference_image,
            "motion_frame_paths": motion_frame_paths,
            "audio_features": audio_features,
            "head_rotation_speeds": head_rotation_speeds
        }

        return sample