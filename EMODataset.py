import os
import json
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from Wav2VecFeatureExtractor import Wav2VecFeatureExtractor
from decord import VideoReader
import decord
from typing import List
from HeadRotation import get_head_pose_velocities_at_frame
from FaceLocator import FaceMaskGenerator 
# Use decord's CPU or GPU context
# For GPU: decord.gpu(0)
decord.logging.set_level(decord.logging.ERROR)
os.environ["OPENCV_LOG_LEVEL"]="FATAL"
import cv2
from typing import List, Tuple, Dict, Any
# from torchvision.transforms.functional import to_tensor
class EmoVideoReader(VideoReader):

    def __init__(self, pixel_transform: transforms.Compose, cond_transform: transforms.Compose, state: torch.Tensor = None):
        super.__init__()
        
        self.pixel_transform = pixel_transform
        self.cond_transform = cond_transform
        self.state = state

    def augmentedImageAtFrame(self, index: int) -> torch.Tensor:

        img = self[index]
        return self.augmentation(img,self.pixel_transform,self.state)
    
    def augmentation(self, images: Any, transform: transforms.Compose, state: torch.Tensor = None) -> torch.Tensor:

            if state is not None:
                torch.set_rng_state(state)
            if isinstance(images, List):
                transformed_images = [transform(img) for img in images]
                ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
            else:
                ret_tensor = transform(images)  # (c, h, w)
            return ret_tensor
    

class EMODataset(Dataset):
    def __init__(self, data_dir: str, sample_rate: int, n_sample_frames: int, width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None):
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

        self.face_mask_generator = FaceMaskGenerator()
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

        decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        self.ctx = decord.gpu(0) # https://github.com/johndpope/decord-cuda12



    def __len__(self) -> int:
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
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_id = self.video_ids[index]
        mp4_path = os.path.join(self.video_dir, f"{video_id}.mp4")

        if self.stage == 'stage1':
            video_reader = VideoReader(mp4_path, ctx=self.ctx)
            video_length = len(video_reader)
            all_frame_images = []
            all_frame_masks = []

            for frame_idx in range(video_length):
     
             # Extract the frame from video_reader, which is a PyTorch tensor
                ref_img_tensor = video_reader[frame_idx]
                                # Add assertions before permute
                assert isinstance(ref_img_tensor, torch.Tensor), "The frame must be a PyTorch tensor."
                assert ref_img_tensor.ndim == 3, "The tensor must be 3-dimensional."
                assert ref_img_tensor.shape[0] == 3 or ref_img_tensor.shape[2] == 3, "The tensor must have 3 channels."

                # Continue with permute and normalization
                ref_img_tensor = ref_img_tensor.permute(2, 0, 1).float() / 255.0  # Convert to CHW and normalize
 
                ref_img_pil = transforms.ToPILImage()(ref_img_tensor.cpu())  
                # Generate the mask using the face mask generator
                mask = self.face_mask_generator.generate_face_region_mask_pil_image(ref_img_pil)
                all_frame_masks.append(mask)
                all_frame_images.append(ref_img_tensor)

            sample = {
                "video_id": video_id,
                "images": all_frame_images,
                "masks": all_frame_masks,
 
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
             
                "audio_features": 0
            }


        elif self.stage == 'stage3':
            # Stage 3: Speed Training
            video_reader = VideoReader(mp4_path, ctx=self.ctx)

            video_length = len(video_reader)
            print("video_length:", video_length)
            print(f"video_id:   https://youtube.com/watch?v={video_id}")
            # Initialize an empty list to collect head rotation speeds for multiple frames
            all_head_rotation_speeds = []

            # Define the number of frames to process from each video
            # For example, process every 10th frame
            frame_step = 10  

            for frame_idx in range(0, video_length, frame_step):
                # Calculate head rotation speeds at the current frame
                head_rotation_speeds = get_head_pose_velocities_at_frame(video_reader, frame_idx, 1)

                # Check if head rotation speeds are successfully calculated
                if head_rotation_speeds:
                    all_head_rotation_speeds.append(head_rotation_speeds)
                else:
                    # Provide a default value if no speeds were calculated
                    #expected_speed_vector_length = 3
                    #default_speeds = torch.zeros(1, expected_speed_vector_length)  # Shape [1, 3]
                    default_speeds = (0.0, 0.0, 0.0)  # List containing one tuple with three elements
                    all_head_rotation_speeds.append(default_speeds)

            
            # Convert list of lists to a tensor
   
            sample = {"all_head_rotation_speeds": all_head_rotation_speeds}
        

            return sample


        return sample
