import os
import torch
import torch.nn as nn
from EmoModel import EMOModel
from FaceMeshMaskGenerator import FaceMeshMaskGenerator
from FramesEncoder import FramesEncoder
from HeadRotation import get_head_pose_velocities
from SpeedEncoder import SpeedEncoder
from VAEEncoder import VAE, ImageEncoder
from EMODataset import EMODataset
from Wav2VecFeatureExtractor import Wav2VecFeatureExtractor
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data  import DataLoader
import cv2
import mediapipe as mp
from diffusers.models.modeling_utils import ModelMixin
import argparse
from omegaconf import OmegaConf

class FaceLocator(ModelMixin):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Define a simple convolutional network to encode the mask
        # Assuming the mask is a 1-channel image (binary mask)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, mask):
        # Assume mask is a 4D tensor with shape (batch_size, channels, height, width)
        return self.encoder(mask)



class VideoEditingModel(ModelMixin):
    def __init__(self, mask_channels, mask_encoder_out_channels):
        super().__init__()
        self.face_locator = FaceLocator(mask_channels, mask_encoder_out_channels)
 

    def forward(self, video_frames, mask):
        # Encode the mask using the Face Locator
        encoded_mask = self.face_locator(mask)
        
        # Add the encoded mask to each video frame's latent representation
        # Assuming video_frames is a tensor with shape (batch_size, channels, num_frames, height, width)
        # and encoded_mask is a tensor with shape (batch_size, channels, height, width)
        # We need to unsqueeze the time dimension for encoded_mask to make it broadcastable
        encoded_mask = encoded_mask.unsqueeze(2)  # Shape: (batch_size, channels, 1, height, width)
        
        # Adding the mask to the video frames' latent representations
        combined_input = video_frames + encoded_mask
        
        # Flatten the temporal dimension into the batch dimension for processing with the backbone
        batch_size, channels, num_frames, height, width = combined_input.shape
        combined_input = combined_input.view(batch_size * num_frames, channels, height, width)
        
        # Pass the combined input through the backbone network
        processed_frames = self.backbone(combined_input)
        
        # Reshape the output to separate the temporal dimension from the batch dimension
        processed_frames = processed_frames.view(batch_size, num_frames, channels, height, width)
        
        return processed_frames





def generate_noisy_latents(vae, timesteps, batch_size, latent_dim, device):
    # Sample latent vectors from the VAE
    latents = torch.randn(batch_size, latent_dim).to(device)
    latents = vae.decode(latents)

    # Add noise to the latents based on the timesteps
    noisy_latents = []
    for t in timesteps:
        noise = torch.randn_like(latents)
        noisy_latent = latents + noise * t
        noisy_latents.append(noisy_latent)

    noisy_latents = torch.stack(noisy_latents, dim=0)
    return noisy_latents


reference_unet_config = {
    "sample_size": 256,                # The size of the input samples
    "in_channels": 3,                  # The number of input channels (e.g., for RGB images this is 3)
    "out_channels": 3,                 # The number of output channels
    "down_block_types": ("DownBlock2D",) * 4,   # A tuple defining the types of blocks in the downsampling path
    "up_block_types": ("UpBlock2D",) * 4,       # A tuple defining the types of blocks in the upsampling path
    # ... Additional configurations
}

denoising_unet_config = {
    "sample_size": 256,                # The size of the input samples
    "in_channels": 3,                  # The number of input channels (e.g., for RGB images this is 3)
    "out_channels": 3,                 # The number of output channels
    "down_block_types": ("DownBlock2D", "AttnDownBlock2D") * 2,   # A tuple defining the types of blocks, including attention blocks
    "up_block_types": ("UpBlock2D", "AttnUpBlock2D") * 2,         # A tuple defining the types of blocks, including attention blocks
    # ... Additional configurations
}

# Configuration for the EMOModel
config = {
    "num_speed_buckets": 10,
    "speed_embedding_dim": 64,
    "reference_unet_config": reference_unet_config,
    "denoising_unet_config": denoising_unet_config,
    # ... Additional model configurations
}



# BACKBONE ~ MagicAnimate class
def main(cfg):
    # Example usage:
    batch_size = 1
    num_frames = 10
    height, width = 256, 256
    mask_channels = 1
    mask_encoder_out_channels = 64

    # Create random video frames and mask
    frames_encoder = FramesEncoder(use_feature_extractor=True)
    #video_frames = torch.rand(batch_size, 3, num_frames, height, width)  # Example video frames tensor
    video_frame_tensor = frames_encoder.encode("images_folder/M2Ohb0FAaJU_1")

    mask = torch.rand(batch_size, mask_channels, height, width)  # Example mask tensor

    # Create the video editing model
    model = VideoEditingModel(mask_channels, mask_encoder_out_channels)

    # Process the video frames with the mask
    output = model(video_frame_tensor, mask)


    # Initialize mediapipe face detection and face mesh models.
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    # Read in the image
    image = cv2.imread('/images_folder/M2Ohb0FAaJU_1/frame_0000.jpg')

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_generator = FaceMeshMaskGenerator()
    mask_tensor = mask_generator.generate_mask(image)


    # Add batch and channel dimensions
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    # Use the mask tensor in your PyTorch model
    # For example:
    # model_output = your_pytorch_model(some_input_tensor, mask_tensor)
    feature_extractor = Wav2VecFeatureExtractor(model_name='facebook/wav2vec2-base-960h', device='cuda')
    video_path = 'images_folder/M2Ohb0FAaJU_1'
    audio_features = feature_extractor.extract_features_from_mp4(video_path, m=2, n=2)


    frames_encoder = FramesEncoder()
    reference_image_path = 'images_folder/M2Ohb0FAaJU_1/frame_0000.jpg'  
    motion_frames_folder = 'images_folder/M2Ohb0FAaJU_1' 

    reference_image_tensor = frames_encoder.encode_reference_image(reference_image_path)
    motion_frames_tensor = frames_encoder.encode_motion_frames(motion_frames_folder)


    # Assuming you have the other required inputs (noisy_latents, timesteps, ref_image, speed_buckets)
    # Instantiate the VAE and image encoder
    latent_dim = 256
    embedding_dim = 512
    vae = VAE(latent_dim)
    image_encoder = ImageEncoder(embedding_dim)

    # Instantiate the EMOModel
    emo_model = EMOModel(vae, image_encoder, config)



    # Specify the necessary parameters
    batch_size = 1
    latent_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_frames = len(os.listdir(motion_frames_folder))


    timesteps = torch.linspace(0, 1, num_frames).to(device)  # Adjust the number of timesteps as needed

    # Generate noisy latents
    noisy_latents = generate_noisy_latents(vae, timesteps, batch_size, latent_dim, device)

    num_speed_buckets = 10
    speed_embedding_dim = 64
    speed_encoder = SpeedEncoder(num_speed_buckets, speed_embedding_dim)

    head_rotation_speeds = get_head_pose_velocities(motion_frames_folder)
    speed_embeddings = speed_encoder(head_rotation_speeds)



    # Create data loaders for each training stage
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])



    stage1_dataset = EMODataset(width=cfg.data.train_width,
            height=cfg.data.train_height,
            n_sample_frames=cfg.data.n_sample_frames,
            sample_rate=cfg.data.sample_rate,
            img_scale=(1.0, 1.0),data_dir='./images_folder', audio_dir='./images_folder', json_file='path/to/celebvhq_info.json', stage='stage1', transform=transform)
    stage1_dataloader = DataLoader(stage1_dataset, batch_size=32, shuffle=True, num_workers=4)

    stage2_dataset = EMODataset(width=cfg.data.train_width,
            height=cfg.data.train_height,
            n_sample_frames=cfg.data.n_sample_frames,
            sample_rate=cfg.data.sample_rate,
            img_scale=(1.0, 1.0),data_dir='./images_folder', audio_dir='./images_folder', json_file='path/to/celebvhq_info.json', stage='stage2', transform=transform)
    stage2_dataloader = DataLoader(stage2_dataset, batch_size=16, shuffle=True, num_workers=4)

    stage3_dataset = EMODataset(width=cfg.data.train_width,
            height=cfg.data.train_height,
            n_sample_frames=cfg.data.n_sample_frames,
            sample_rate=cfg.data.sample_rate,
            img_scale=(1.0, 1.0),data_dir='./images_folder', audio_dir='./images_folder', json_file='path/to/celebvhq_info.json', stage='stage3', transform=transform)
    stage3_dataloader = DataLoader(stage3_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(emo_model.parameters(), lr=0.001)
    output = emo_model(noisy_latents, timesteps, reference_image_tensor, motion_frames_tensor, audio_features, speed_embeddings)



    ## DRAFT - NOT FINAL

    num_epochs_stage1 = 20
    num_epochs_stage2 = 20
    num_epochs_stage3 = 20

    mask_generator = FaceMeshMaskGenerator()

    # Stage 1: Image Pretraining
    # The first stage is the image pretraining, where the Backbone Network,
    # the ReferenceNet, and the Face Locator are token into training, in this stage,
    # the Backbone takes a single frame as input, while ReferenceNet handles a distinct,
    # randomly chosen frame from the same video clip
    for epoch in range(num_epochs_stage1):
        for batch in stage1_dataloader:
            reference_image = batch['reference_image']
            backbone_image = batch['backbone_image']

            
            # Forward pass through ReferenceNet
            reference_output = reference_network(reference_image)
            
            # Forward pass through Face Locator
            face_region_mask = mask_generator(backbone_image)
            
            # Calculate loss and perform optimization
            loss = criterion(backbone_output, reference_output)  # Define appropriate loss function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    
    config = OmegaConf.load("./configs/training/stage2.yaml")
    
    # Stage 2: Video Training
    for epoch in range(num_epochs_stage2):
        for batch in stage2_dataloader:
            motion_frames = batch['motion_frames']
            video_frames = batch['video_frames']
            audio_features = batch['audio_features']
            
            # Forward pass through Backbone Network
            backbone_output = backbone_network(video_frames)
            
            # Forward pass through ReferenceNet
            reference_output = reference_network(motion_frames)
            
            # Forward pass through Face Locator
            face_region_mask = face_locator(video_frames)
            
            # Forward pass through temporal modules
            temporal_output = temporal_modules(backbone_output)
            
            # Forward pass through audio layers
            audio_output = audio_layers(audio_features)
            
            # Calculate loss and perform optimization
            loss = criterion(temporal_output, audio_output)  # Define appropriate loss function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Stage 3: Speed Training
    for epoch in range(num_epochs_stage3):
        for batch in stage3_dataloader:
            video_frames = batch['video_frames']
            head_rotation_speeds = batch['head_rotation_speeds']
            
            # Forward pass through Backbone Network
            backbone_output = backbone_network(video_frames)
            
            # Forward pass through temporal modules
            temporal_output = temporal_modules(backbone_output)
            
            # Forward pass through speed layers
            speed_output = speed_layers(head_rotation_speeds)
            
            # Calculate loss and perform optimization
            loss = criterion(temporal_output, speed_output)  # Define appropriate loss function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1.yaml")
    main(config)
