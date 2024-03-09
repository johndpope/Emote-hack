import os
import json
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from Wav2VecFeatureExtractor import Wav2VecFeatureExtractor
from HeadRotation import get_head_pose_velocities
from decord import VideoReader
from typing import List
import numpy as np

class EMODataset(Dataset):
    def __init__(self, data_dir,sample_rate,n_sample_frames, width,  height,  img_scale=(1.0, 1.0),  img_ratio=(0.9, 1.0),  drop_ratio=0.1, audio_dir, json_file, stage='stage1', transform=None):
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.transform = transform
        self.stage = stage
        self.feature_extractor = Wav2VecFeatureExtractor(model_name='facebook/wav2vec2-base-960h', device='cuda')


        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio
        with open(json_file, 'r') as f:
            self.celebvhq_info = json.load(f)

        self.video_ids = list(self.celebvhq_info['clips'].keys())

    def __len__(self):
        return len(self.video_ids)

    def augmentation(self, images, transform, state=None):
            if state is not None:
                torch.set_rng_state(state)
            if isinstance(images, List):
                transformed_images = [transform(img) for img in images]
                ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
            else:
                ret_tensor = transform(images)  # (c, h, w)
            return ret_tensor
    
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video_info = self.celebvhq_info['clips'][video_id]
        ytb_id = video_info['ytb_id']

        frame_folder = os.path.join(self.data_dir, f"{ytb_id}_{video_id.split('_')[-1]}")
        mp4_path = os.path.join(self.audio_dir, f"{ytb_id}_{video_id.split('_')[-1]}.mp4")
        frames = sorted([frame for frame in os.listdir(frame_folder) if frame.endswith(".jpg")])

        if self.stage == 'stage1':
            # Stage 1: Image Pretraining

            video_reader = VideoReader(mp4_path)
            video_length = len(video_reader)
            rnd_idx = random.randint(0, video_length - clip_length)
            ref_img = Image.fromarray(video_reader[rnd_idx].asnumpy())
            #reference_frame = frames[0]  # Use the first frame as the reference frame
            backbone_frame = random.choice(frames)  # Randomly select a frame for the Backbone Network

         
            backbone_image = Image.open(os.path.join(frame_folder, backbone_frame))

            if self.transform:
                ref_img = self.transform(ref_img)
                backbone_image = self.transform(backbone_image)

            sample = {
                "video_id": video_id,
                "reference_image": ref_img,
                "backbone_image": backbone_image
            }

        elif self.stage == 'stage2':
            # Stage 2: Video Training
            motion_frames = frames[:4]  # Use the first 4 frames as motion frames
            video_frames = frames[4:8]  # Use the next 4 frames for video training

            video_reader = VideoReader(mp4_path)
            motion_frame_paths = [os.path.join(frame_folder, frame) for frame in motion_frames]
            video_frame_paths = [os.path.join(frame_folder, frame) for frame in video_frames]

            motion_images = [Image.open(path) for path in motion_frame_paths]
            video_images = [Image.open(path) for path in video_frame_paths]

            if self.transform:
                motion_images = [self.transform(image) for image in motion_images]
                video_images = [self.transform(image) for image in video_images]

            audio_features = self.feature_extractor.extract_features_from_mp4(mp4_path, m=2, n=2)

            video_length = len(video_reader)

            clip_length = min(
                video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
            )
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(
                start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
            ).tolist()

            # read frames
            vid_pil_image_list = []
            for index in batch_index:
                img = video_reader[index]
                vid_pil_image_list.append(Image.fromarray(img.asnumpy()))


            ref_img_idx = random.randint(0, video_length - 1)
            ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

            # transform
            state = torch.get_rng_state()
            pixel_values_vid = self.augmentation(
                vid_pil_image_list, self.pixel_transform, state
            )
           
            pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
            
            sample = {
                "video_id": video_id,
                "motion_frames": motion_images,
                "video_frames": video_images,
                "audio_features": audio_features,
                "pixel_values_vid": pixel_values_vid,
                "pixel_values_ref_img": pixel_values_ref_img
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
