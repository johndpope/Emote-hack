import torch.nn as nn
from SpeedEncoder import SpeedEncoder
from magicanimate.models.unet_controlnet import  UNet3DConditionModel
from magicanimate.models.controlnet import UNet2DConditionModel

from diffusers.models.modeling_utils import ModelMixin
from Wav2VecFeatureExtractor import Wav2VecFeatureExtractor

import torch
import torch.nn as nn
import torch.nn.functional as F



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
    


