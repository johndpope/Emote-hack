import os
import json
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from Wav2VecFeatureExtractor import Wav2VecFeatureExtractor
from decord import VideoReader
from typing import List
from HeadRotation import get_head_pose_velocities_at_frame


class EmoVideoReader(VideoReader):

    def __init__(self,pixel_transform,cond_transform,state=None):
        super.__init__()
        
        self.pixel_transform = pixel_transform
        self.cond_transform = cond_transform
        self.state = state

    def augmentedImageAtFrame(self,index):

        img = self[index]
        return self.augmentation(img,self.pixel_transform,self.state)
    
    def augmentation(self, images, transform, state=None):
            if state is not None:
                torch.set_rng_state(state)
            if isinstance(images, List):
                transformed_images = [transform(img) for img in images]
                ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
            else:
                ret_tensor = transform(images)  # (c, h, w)
            return ret_tensor
    

class EMODataset(Dataset):
    def __init__(self, data_dir,sample_rate,n_sample_frames, width,  height,  img_scale=(1.0, 1.0),  img_ratio=(0.9, 1.0), video_dir =".", drop_ratio=0.1,  json_file="", stage='stage1', transform=None):
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.video_dir = video_dir
        self.data_dir = data_dir
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
        # video_info = self.celebvhq_info['clips'][video_id]


        # print("video_id:", video_id, "video_info:", video_info)
        # ytb_id = video_info['ytb_id']


        mp4_path = os.path.join(self.video_dir, f"{video_id}.mp4")


        if self.stage == 'stage1':
            # Stage 1: Image Pretraining

            # video_reader = VideoReader(mp4_path)
            # video_length = len(video_reader)
            # rnd_idx = random.randint(0, video_length-1)
            # ref_img = Image.fromarray(video_reader[rnd_idx].asnumpy())
            #reference_frame = frames[0]  # Use the first frame as the reference frame
   
         
         #   backbone_image = Image.open(os.path.join(frame_folder, backbone_frame))

            # if self.transform:
            #     ref_img = self.transform(ref_img)
            #     backbone_image = self.transform(backbone_image)

            sample = {
                "video_id": video_id,
                "reference_image": 0,
                "backbone_image": []
            }

        elif self.stage == 'stage2':
            # Stage 2: Video Training

         #   video_reader = VideoReader(mp4_path)
            
            # # Extract audio features for the specific frame
            # audio_features = self.feature_extractor.extract_features_for_frame(mp4_path, backbone_frame, m=2)

            # ref_img = Image.fromarray(video_reader[backbone_frame].asnumpy())

            # transform
            # state = torch.get_rng_state()
            # pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
            
            sample = {
                "f_idx" : 0,
                "video_id": video_id,
             
                "audio_features": 0,
                # "pixel_values_ref_img": pixel_values_ref_img
            }

        elif self.stage == 'stage3':
            # Stage 3: Speed Training
            video_reader = VideoReader(mp4_path)
            video_length = len(video_reader)
     

            rnd_idx = random.randint(0, video_length-1)
            head_rotation_speeds = get_head_pose_velocities_at_frame(video_reader,rnd_idx ,1)
            #print("head_rotation_speeds:",head_rotation_speeds)
            sample = {
                "head_rotation_speeds": head_rotation_speeds,
            }

        return sample
