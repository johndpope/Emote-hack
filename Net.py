import torch
import torch.nn as nn
import torch.nn.functional as F
from magicanimate.models.unet import UNet3DConditionModel
from diffusers.models.modeling_utils import ModelMixin
from magicanimate.models.unet_controlnet import  UNet3DConditionModel
from magicanimate.models.controlnet import UNet2DConditionModel

import json
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader
import decord
from typing import List
from HeadRotation import get_head_pose_velocities_at_frame
from torchvision.transforms import ToTensor
import numpy as np
from moviepy.editor import VideoFileClip
import os
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import cv2
import mediapipe as mp



# JAM EVERYTHING INTO 1 CLASS - so Claude 3 / Chatgpt can analyze at once
class Encoder(nn.Module):
    """
    Encoder encodes both reference and motion frames using a shared VAE encoder structure. 
    This simulates the behavior depicted in the Frames Encoding section of the diagram, 
    which indicates that reference and motion frames are passed through the same encoder.
    """
    def __init__(self, input_channels, latent_dim, img_size):
        super(Encoder, self).__init__()
        # Define the layers for the encoder according to the VAE structure.
        self.encoder_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * (img_size // 16) * (img_size // 16), latent_dim * 2),
        )
    
    def forward(self, x):
        # Pass input through encoder layers to get the latent space representation
        latent = self.encoder_layers(x)
        # Split the result into mu and logvar components of the latent space
        mu, logvar = torch.chunk(latent, 2, dim=-1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels,img_size):
        super(Decoder, self).__init__()
        self.img_size = img_size
        # The output size of the last deconvolution would be [output_channels, img_size, img_size]
        self.fc = nn.Linear(latent_dim, 256 * (img_size // 16) * (img_size // 16))
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 256, self.img_size // 16, self.img_size // 16)  # Reshape z to (batch_size, 256, img_size/16, img_size/16)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        reconstruction = torch.sigmoid(self.deconv4(z))  # Use sigmoid for normalizing the output to [0, 1]
        return reconstruction




class FramesEncodingVAE(nn.Module):
    """
    FramesEncodingVAE combines the encoding of reference and motion frames with additional components
    such as ReferenceNet and SpeedEncoder as depicted in the Frames Encoding part of the diagram.
    """
    def __init__(self, input_channels, latent_dim, img_size, reference_net):
        super(FramesEncodingVAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim, img_size)
        self.decoder = Decoder(latent_dim, input_channels, img_size)
        self.reference_net = reference_net

        # SpeedEncoder can be implemented as needed.
        self.speed_encoder = nn.Sequential(
            # Dummy layers for illustrative purposes; replace with actual speed encoding mechanism.
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, reference_image, motion_frames, speed_value):
        # Encode reference and motion frames
        reference_mu, reference_logvar = self.encoder(reference_image)
        motion_mu, motion_logvar = self.encoder(motion_frames)

        # Reparameterize reference and motion latent vectors
        reference_z = self.reparameterize(reference_mu, reference_logvar)
        motion_z = self.reparameterize(motion_mu, motion_logvar)

        # Process reference features with ReferenceNet
        reference_features = self.reference_net(reference_z)

        # Embed speed value
        speed_embedding = self.speed_encoder(speed_value)

        # Combine features
        combined_features = torch.cat([reference_features, motion_z, speed_embedding], dim=1)

        # Decode the combined features
        reconstructed_frames = self.decoder(combined_features)

        return reconstructed_frames, reference_mu, reference_logvar, motion_mu, motion_logvar

    def vae_loss(self, recon_frames, reference_image, motion_frames, reference_mu, reference_logvar, motion_mu, motion_logvar):
        # Reconstruction loss (MSE or BCE, depending on the final activation of the decoder)
        recon_loss = F.mse_loss(recon_frames, torch.cat([reference_image, motion_frames], dim=1), reduction='sum')
        
        # KL divergence loss for reference and motion latent vectors
        kl_loss_reference = -0.5 * torch.sum(1 + reference_logvar - reference_mu.pow(2) - reference_logvar.exp())
        kl_loss_motion = -0.5 * torch.sum(1 + motion_logvar - motion_mu.pow(2) - motion_logvar.exp())
        
        return recon_loss + kl_loss_reference + kl_loss_motion
    




    
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ReferenceNet(nn.Module):
    def __init__(self, vae_model,  speed_encoder,config):
        super(ReferenceNet, self).__init__()
        # Define the number of input channels and the scaling factor for feature channels
        num_channels = 3  # For RGB images
        feature_scale = 64  # Example scaling factor

        # Reference UNet (ReferenceNet)
        self.reference_unet = UNet3DConditionModel(**config.reference_unet_config)
        # Initialize the components
        self.vae = vae_model

        self.speed_encoder = speed_encoder

        # Downsample and Upsample Blocks
        self.down1 = DownsampleBlock(num_channels, feature_scale)
        self.down2 = DownsampleBlock(feature_scale, feature_scale * 2)
        self.down3 = DownsampleBlock(feature_scale * 2, feature_scale * 4)
        self.up1 = UpsampleBlock(feature_scale * 4, feature_scale * 2)
        self.up2 = UpsampleBlock(feature_scale * 2, feature_scale)

        # Final convolution to adjust the number of output channels
        self.final_conv = nn.Conv2d(feature_scale, num_channels, kernel_size=1)

    def forward(self, reference_image, motion_frames, head_rotation_speed):
        # Downsample reference image
        ref_x1 = self.down1(reference_image)
        ref_x2 = self.down2(ref_x1)
        ref_x3 = self.down3(ref_x2)

        # Pass motion frames through similar downsampling blocks
        motion_x = motion_frames.view(-1, motion_frames.size(2), motion_frames.size(3), motion_frames.size(4))
        motion_x1 = self.down1(motion_x)
        motion_x2 = self.down2(motion_x1)
        motion_x3 = self.down3(motion_x2)

        # Upsample and integrate features from motion frames
        x = self.up1(ref_x3, motion_x3)
        x = self.up2(x, ref_x2)

        # Final convolution to adjust the number of output channels
        out = self.final_conv(x)

        # Pass the output through 
        reference_features = self.reference_unet(out)

        # Encode speed and expand its dimensions to concatenate with reference features
        speed_embedding = self.speed_encoder(head_rotation_speed)
        speed_embedding = speed_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, reference_features.size(2), reference_features.size(3))

        # Combine reference features and speed embedding
        combined_features = torch.cat([reference_features, speed_embedding], dim=1)

        return combined_features


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
        # This method is now designed to handle a tensor of head rotation speeds
        # head_rotation_speed should be a 1D tensor of shape (batch_size,)
        assert head_rotation_speed.ndim == 1, "head_rotation_speed must be a 1D tensor"

        # Initialize a tensor to hold the encoded speed vectors
        speed_vectors = torch.zeros((head_rotation_speed.size(0), self.num_speed_buckets), dtype=torch.float32)

        for i in range(self.num_speed_buckets):
            center = self.bucket_centers[i]
            radius = self.bucket_radii[i]

            # Element-wise operation to compute the tanh encoding for each speed value in the batch
            speed_vectors[:, i] = torch.tanh((head_rotation_speed - center) / radius * 3)

        return speed_vectors

    def forward(self, head_rotation_speeds):
        # Ensure that head_rotation_speeds is a 1D Tensor of floats
        assert head_rotation_speeds.ndim == 1, "head_rotation_speeds must be a 1D tensor"
        assert head_rotation_speeds.dtype == torch.float32, "head_rotation_speeds must be a tensor of floats"

        # Process the batch of head rotation speeds through the encoder
        speed_vectors = self.encode_speed(head_rotation_speeds)

        # Pass the encoded vectors through the MLP
        speed_embeddings = self.mlp(speed_vectors)
        return speed_embeddings
    



class CrossAttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttentionLayer, self).__init__()
        # Assuming feature_dim is the dimensionality of the features from the audio encoder and the Backbone Network

        # Query, Key, Value transformations
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        # Scaling factor for the dot product
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))

    def forward(self, latent_code, audio_features):
        """
        latent_code: the visual feature maps from the Backbone Network
        audio_features: the extracted audio features from the audio encoder

        Returns:
        The output after applying cross attention.
        """
        # Generate query, key, value vectors
        assert latent_code.dim() == 3, "Expected latent_code to be a 3D tensor (batch, features, seq_len)"
        assert audio_features.dim() == 3, "Expected audio_features to be a 3D tensor (batch, features, seq_len)"
        assert latent_code.size(1) == audio_features.size(1), "Feature dimensions of latent_code and audio_features must match"

        query = self.query(latent_code)
        key = self.key(audio_features)
        value = self.value(audio_features)

        # Compute the attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Apply softmax to get probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply the attention to the values
        attention_output = torch.matmul(attention_probs, value)

        return attention_output

class AudioAttentionLayers(nn.Module):
    def __init__(self, feature_dim, num_layers):
        super(AudioAttentionLayers, self).__init__()
        assert feature_dim > 0, "Feature dimension must be positive"
        assert num_layers > 0, "Number of layers must be positive"
        self.layers = nn.ModuleList([CrossAttentionLayer(feature_dim) for _ in range(num_layers)])

    def forward(self, latent_code, audio_features):
        """
        latent_code: the visual feature maps from the Backbone Network
        audio_features: the extracted audio features from the audio encoder

        Returns:
        The combined output after applying all audio-attention layers.
        """
        assert latent_code.dim() == 3, "Expected latent_code to be a 3D tensor (batch, features, seq_len)"
        assert audio_features.dim() == 3, "Expected audio_features to be a 3D tensor (batch, features, seq_len)"

        for layer in self.layers:
            latent_code = layer(latent_code, audio_features) + latent_code  # Adding skip-connection

        return latent_code



class EMOModel(ModelMixin):
    def __init__(self, vae, image_encoder, config):
        super().__init__()
        self.vae = vae
        self.image_encoder = image_encoder

        # Reference UNet (ReferenceNet)
        self.reference_unet = UNet2DConditionModel(**config.reference_unet_config)


        # Integrate Wav2Vec Feature Extractor
        self.wav2vec_feature_extractor = Wav2VecFeatureExtractor(pretrained_model_name="wav2vec_model_name")

        # Denoising UNet (Backbone Network)
        # Ensure it integrates ReferenceNet and audio features
        self.denoising_unet = UNet3DConditionModel(
            sample_size=config.denoising_unet_config.get("sample_size"),
            in_channels=config.denoising_unet_config.get("in_channels"),
            out_channels=config.denoising_unet_config.get("out_channels"),
            down_block_types=config.denoising_unet_config.get("down_block_types"),
            up_block_types=config.denoising_unet_config.get("up_block_types"),
            block_out_channels=config.denoising_unet_config.get("block_out_channels"),
            layers_per_block=config.denoising_unet_config.get("layers_per_block"),
            cross_attention_dim=config.denoising_unet_config.get("cross_attention_dim"),
            attention_head_dim=config.denoising_unet_config.get("attention_head_dim"),
            use_motion_module=True,
            motion_module_type='simple',
            motion_module_kwargs=config.denoising_unet_config.get("motion_module_kwargs"),
        )

        # Face Region Controller
        self.face_locator = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Speed Controller
        self.speed_encoder = SpeedEncoder(config.num_speed_buckets, config.speed_embedding_dim)

        config = {
            "audio_feature_dim": 768,
            "audio_num_layers": 3
        }
        # Audio Attention Layers
        self.audio_attention_layers = AudioAttentionLayers(feature_dim=config["audio_feature_dim"], num_layers=config["audio_num_layers"])


    def forward(self, noisy_latents, timesteps, ref_image, motion_frames, audio_features, head_rotation_speeds):
            batch_size, num_frames, _, height, width = noisy_latents.shape



            # Encode reference image
            ref_image_latents = self.vae.encode(ref_image).latent_dist.sample()
            ref_image_latents = ref_image_latents * 0.18215
            ref_image_embeds = self.image_encoder(ref_image)

            # Encode motion frames
            motion_frames_latents = self.vae.encode(motion_frames).latent_dist.sample()
            motion_frames_latents = motion_frames_latents * 0.18215
            motion_frames_embeds = self.image_encoder(motion_frames)

            # Process audio input through Wav2Vec feature extractor
            # Get audio embeddings from the extracted features
            audio_embeds = self.wav2vec_feature_extractor(audio_features)



            # Compute face region mask
            face_region_mask = self.face_locator(ref_image)
            face_region_mask = face_region_mask.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

            # Forward pass through Reference UNet
            ref_embeds = self.reference_unet(ref_image_latents, timesteps, ref_image_embeds).sample

            # Get speed embeddings from the head rotation speeds
            speed_embeddings = self.speed_encoder(head_rotation_speeds)



            # Apply audio attention layers
            # NOTE: You might need to modify the denoising_unet to return intermediate latent code 
            #       before final output for this step.
            intermediate_latent_code = self.denoising_unet.get_intermediate_latent_code(noisy_latents, timesteps)
            latent_code_with_audio = self.audio_attention_layers(intermediate_latent_code, audio_embeds)


            # Forward pass through Denoising UNet (Backbone Network)
            # Incorporate the latent_code_with_audio into the denoising process
            # The modified latent code is now used in the forward pass of the Denoising UNet
            model_pred = self.denoising_unet(
                sample=latent_code_with_audio,  # Using latent code modified by audio attention
                timestep=timesteps,
                encoder_hidden_states=ref_embeds,
                motion_fea=motion_frames_embeds,
                audio_fea=audio_embeds,  # original audio features can still be used in parallel if needed
                speed_fea=speed_embeddings,
                pose_cond_fea=face_region_mask,
            ).sample

            return model_pred
        


class Wav2VecFeatureExtractor:
    def __init__(self, model_name='facebook/wav2vec2-base-960h', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)

    def extract_features_from_mp4(self, video_path, m=2, n=2):
        """
        Extract audio features from an MP4 file using Wav2Vec 2.0.

        Args:
            video_path (str): Path to the MP4 video file.
            m (int): The number of frames before the current frame to include.
            n (int): The number of frames after the current frame to include.

        Returns:
            torch.Tensor: Features extracted from the audio for each frame.
        """
        # Create the audio file path from the video file path
        audio_path = os.path.splitext(video_path)[0] + '.wav'

        # Check if the audio file already exists
        if not os.path.exists(audio_path):
            # Extract audio from video
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio_path)

        # Load the audio file
        waveform, sample_rate = sf.read(audio_path)

        # Check if we need to resample
        if sample_rate != self.processor.feature_extractor.sampling_rate:
            waveform = librosa.resample(np.float32(waveform), orig_sr=sample_rate, target_sr=self.processor.feature_extractor.sampling_rate)
            sample_rate = self.processor.feature_extractor.sampling_rate

        # Ensure waveform is a 1D array for a single-channel audio
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # Taking mean across channels for simplicity

        # Process the audio to extract features
        input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
        input_values = input_values.to(self.device)

        # Pass the input_values to the model
        with torch.no_grad():
            hidden_states = self.model(input_values).last_hidden_state

        num_frames = hidden_states.shape[1]
        feature_dim = hidden_states.shape[2]

        # Concatenate nearby frame features
        all_features = []
        for f in range(num_frames):
            start_frame = max(f - m, 0)
            end_frame = min(f + n + 1, num_frames)
            frame_features = hidden_states[0, start_frame:end_frame, :].flatten()

            # Add padding if necessary
            if f - m < 0:
                front_padding = torch.zeros((m - f) * feature_dim, device=self.device)
                frame_features = torch.cat((front_padding, frame_features), dim=0)
            if f + n + 1 > num_frames:
                end_padding = torch.zeros(((f + n + 1 - num_frames) * feature_dim), device=self.device)
                frame_features = torch.cat((frame_features, end_padding), dim=0)

            all_features.append(frame_features)

        all_features = torch.stack(all_features, dim=0)
        return all_features
    



    def extract_features_for_frame(self, video_path, frame_index, m=2):
        """
        Extract audio features for a specific frame from an MP4 file using Wav2Vec 2.0.

        Args:
            video_path (str): Path to the MP4 video file.
            frame_index (int): The index of the frame to extract features for.
            m (int): The number of frames before and after the current frame to include.

        Returns:
            torch.Tensor: Features extracted from the audio for the specified frame.
        """
        # Create the audio file path from the video file path
        audio_path = os.path.splitext(video_path)[0] + '.wav'

        # Check if the audio file already exists
        if not os.path.exists(audio_path):
            # Extract audio from video
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio_path)

        # Load the audio file
        waveform, sample_rate = sf.read(audio_path)

        # Check if we need to resample
        if sample_rate != self.processor.feature_extractor.sampling_rate:
            waveform = librosa.resample(np.float32(waveform), orig_sr=sample_rate, target_sr=self.processor.feature_extractor.sampling_rate)
            sample_rate = self.processor.feature_extractor.sampling_rate

        # Ensure waveform is a 1D array for a single-channel audio
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # Taking mean across channels for simplicity

        # Process the audio to extract features
        input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
        input_values = input_values.to(self.device)

        # Pass the input_values to the model
        with torch.no_grad():
            hidden_states = self.model(input_values).last_hidden_state

        num_frames = hidden_states.shape[1]
        feature_dim = hidden_states.shape[2]

        # Concatenate nearby frame features
        all_features = []
        start_frame = max(frame_index - m, 0)
        end_frame = min(frame_index + m + 1, num_frames)
        frame_features = hidden_states[0, start_frame:end_frame, :].flatten()
        
        # Add padding if necessary
        if frame_index - m < 0:
            front_padding = torch.zeros((m - frame_index) * feature_dim, device=self.device)
            frame_features = torch.cat((front_padding, frame_features), dim=0)
        if frame_index + m + 1 > num_frames:
            end_padding = torch.zeros(((frame_index + m + 1) - num_frames) * feature_dim, device=self.device)
            frame_features = torch.cat((frame_features, end_padding), dim=0)
        
        all_features.append(frame_features)
        
        return torch.stack(all_features)
    
    # This is a dummy example of a neural network module that might take the concatenated frame features
class AudioFeatureModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(AudioFeatureModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)
    


# given an image - spit out the mask

# Instantiate the model
# model = FaceLocator()

# Assuming 'input_image' is a torch tensor of shape (B, C, H, W)
# Get the binary mask output from the model
# binary_mask = model(input_image)

class FaceLocator(nn.Module):
    def __init__(self):
        super(FaceLocator, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Define the final convolutional layer that outputs a single channel (mask)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, images):
        # Forward pass through the convolutional layers
        # Assert that images are of the correct type (floating-point)
        assert images.dtype == torch.float32, 'Images must be of type torch.float32'
        # Assert that images have 4 dimensions [B, C, H, W]
        assert images.ndim == 4, 'Images must have 4 dimensions [B, C, H, W]'

        x = F.relu(self.conv1(images))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # Shape after pooling: (B, 64, H/8, W/8)
        

        assert x.size(1) == 64, f"Input to final conv layer has {x.size(1)} channels, expected 64."

        # Pass through the final convolutional layer to get a single channel output
        logits = self.final_conv(x)  # Output logits directly, Shape: (B, 1, H/8, W/8)
        
        # No sigmoid or thresholding here because BCEWithLogitsLoss will handle it

        # Upsample logits to the size of the original image
        logits_upsampled = F.interpolate(logits, size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
        
        return logits_upsampled





class FaceMaskGenerator:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        # Initialize FaceDetection once here
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    def __del__(self):
        self.face_detection.close()
        self.face_mesh.close()

    def generate_face_region_mask(self,frame_image, video_id=0,frame_idx=0):
        frame_np = np.array(frame_image.convert('RGB'))  # Ensure the image is in RGB
        return self.generate_face_region_mask_np_image(video_id,frame_idx,frame_np)

    def generate_face_region_mask_np_image(self,frame_np, video_id=0,frame_idx=0, padding=10):
        # Convert from RGB to BGR for MediaPipe processing
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        height, width, _ = frame_bgr.shape

        # Create a blank mask with the same dimensions as the frame
        mask = np.zeros((height, width), dtype=np.uint8)

        # Optionally save a debug image
        debug_image = mask
        # Detect faces
        detection_results = self.face_detection.process(frame_bgr)
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * width)
                ymin = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                # Draw a rectangle on the debug image for each detection
                cv2.rectangle(debug_image, (xmin, ymin), (xmin + bbox_width, ymin + bbox_height), (0, 255, 0), 2)
        # Check that detections are not None
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * width)
                ymin = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                # Calculate padded coordinates
                pad_xmin = max(0, xmin - padding)
                pad_ymin = max(0, ymin - padding)
                pad_xmax = min(width, xmin + bbox_width + padding)
                pad_ymax = min(height, ymin + bbox_height + padding)

                # Draw a white padded rectangle on the mask
                mask[pad_ymin:pad_ymax, pad_xmin:pad_xmax] = 255

               
                # cv2.rectangle(debug_image, (pad_xmin, pad_ymin), 
                            #   (pad_xmax, pad_ymax), (255, 255, 255), thickness=-1)
                # cv2.imwrite(f'./temp/debug_face_mask_{video_id}-{frame_idx}.png', debug_image)

        return mask

    
    def generate_face_region_mask_pil_image(self,frame_image,video_id=0, frame_idx=0):
        # Convert from PIL Image to NumPy array in BGR format
        frame_np = np.array(frame_image.convert('RGB'))  # Ensure the image is in RGB
        return self.generate_face_region_mask_np_image(frame_np,video_id,frame_idx,)
    
    import os


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
    def __init__(self, use_gpu:False,data_dir: str, sample_rate: int, n_sample_frames: int, width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None):
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
        self.use_gpu = use_gpu

        decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        self.ctx = decord.cpu()


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
            
            transform_to_tensor = ToTensor()
            # Read frames and generate masks
            vid_pil_image_list = []
            mask_tensor_list = []
            for frame_idx in range(video_length):
                # Read frame and convert to PIL Image
                frame = Image.fromarray(video_reader[frame_idx].numpy())

                # Transform the frame
                state = torch.get_rng_state()
                pixel_values_frame = self.augmentation(frame, self.pixel_transform, state)
                vid_pil_image_list.append(pixel_values_frame)


                # Convert the transformed frame back to NumPy array in RGB format
                transformed_frame_np = np.array(pixel_values_frame.permute(1, 2, 0).numpy() * 255, dtype=np.uint8)
                transformed_frame_np = cv2.cvtColor(transformed_frame_np, cv2.COLOR_RGB2BGR)

                # Generate the mask using the face mask generator
                mask_np = self.face_mask_generator.generate_face_region_mask_np_image(transformed_frame_np, video_id, frame_idx)

                    # Convert the mask from numpy array to PIL Image
                mask_pil = Image.fromarray(mask_np)

                # Transform the PIL Image mask to a PyTorch tensor
                mask_tensor = transform_to_tensor(mask_pil)
                mask_tensor_list.append(mask_tensor)

            sample = {
                "video_id": video_id,
                "images": vid_pil_image_list,
                "masks": mask_tensor_list,
               
            }


        elif self.stage == 'stage2':
            # Stage 2: Video Training

            
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
