import os
import json
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from Wav2VecFeatureExtractor import Wav2VecFeatureExtractor
from HeadRotation import get_head_pose_velocities

class EMODataset(Dataset):
    def __init__(self, data_dir, audio_dir, json_file, stage='stage1', transform=None):
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.transform = transform
        self.stage = stage
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
        mp4_path = os.path.join(self.audio_dir, f"{ytb_id}_{video_id.split('_')[-1]}.mp4")
        frames = sorted([frame for frame in os.listdir(frame_folder) if frame.endswith(".jpg")])

        if self.stage == 'stage1':
            # Stage 1: Image Pretraining
            reference_frame = frames[0]  # Use the first frame as the reference frame
            backbone_frame = random.choice(frames)  # Randomly select a frame for the Backbone Network

            reference_image = Image.open(os.path.join(frame_folder, reference_frame))
            backbone_image = Image.open(os.path.join(frame_folder, backbone_frame))

            if self.transform:
                reference_image = self.transform(reference_image)
                backbone_image = self.transform(backbone_image)

            sample = {
                "video_id": video_id,
                "reference_image": reference_image,
                "backbone_image": backbone_image
            }

        elif self.stage == 'stage2':
            # Stage 2: Video Training
            motion_frames = frames[:4]  # Use the first 4 frames as motion frames
            video_frames = frames[4:8]  # Use the next 4 frames for video training

            motion_frame_paths = [os.path.join(frame_folder, frame) for frame in motion_frames]
            video_frame_paths = [os.path.join(frame_folder, frame) for frame in video_frames]

            motion_images = [Image.open(path) for path in motion_frame_paths]
            video_images = [Image.open(path) for path in video_frame_paths]

            if self.transform:
                motion_images = [self.transform(image) for image in motion_images]
                video_images = [self.transform(image) for image in video_images]

            audio_features = self.feature_extractor.extract_features_from_mp4(mp4_path, m=2, n=2)

            sample = {
                "video_id": video_id,
                "motion_frames": motion_images,
                "video_frames": video_images,
                "audio_features": audio_features
            }

        elif self.stage == 'stage3':
            # Stage 3: Speed Training
            video_frames = frames[4:8]  # Use the same frames as in Stage 2

            video_frame_paths = [os.path.join(frame_folder, frame) for frame in video_frames]
            video_images = [Image.open(path) for path in video_frame_paths]

            if self.transform:
                video_images = [self.transform(image) for image in video_images]

            head_rotation_speeds = get_head_pose_velocities(video_frame_paths)

            sample = {
                "video_id": video_id,
                "video_frames": video_images,
                "head_rotation_speeds": head_rotation_speeds
            }

        return sample
