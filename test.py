import torch
from Net import CrossAttentionLayer,AudioAttentionLayers,ReferenceAttentionLayer,BackboneNetwork, ReferenceNet, AudioAttentionLayers, TemporalModule,FramesEncodingVAE
from diffusers import AutoencoderKL, DDIMScheduler
from Net import EMOModel, VAE, ImageEncoder
import unittest
from Net import SpeedEncoder
import unittest
import torch
from Net import CrossAttentionLayer, AudioAttentionLayers, ReferenceAttentionLayer, BackboneNetwork, ReferenceNet, TemporalModule, FramesEncodingVAE, EMOModel, VAE, ImageEncoder

class TestCrossAttentionLayer(unittest.TestCase):
    def test_output_shape(self):
        feature_dim = 512
        batch_size = 2
        seq_len = 10
        latent_code = torch.randn(batch_size, feature_dim, seq_len)
        audio_features = torch.randn(batch_size, feature_dim, seq_len)
        cross_attention_layer = CrossAttentionLayer(feature_dim)
        output = cross_attention_layer(latent_code, audio_features)
        self.assertEqual(output.shape, (batch_size, feature_dim, seq_len))

class TestAudioAttentionLayers(unittest.TestCase):
    def test_output_shape(self):
        feature_dim = 512
        num_layers = 3
        batch_size = 2
        seq_len = 10
        latent_code = torch.randn(batch_size, feature_dim, seq_len)
        audio_features = torch.randn(batch_size, feature_dim, seq_len)
        audio_attention_layers = AudioAttentionLayers(feature_dim, num_layers)
        output = audio_attention_layers(latent_code, audio_features)
        self.assertEqual(output.shape, (batch_size, feature_dim, seq_len))

class TestReferenceAttentionLayer(unittest.TestCase):
    def test_output_shape(self):
        feature_dim = 512
        batch_size = 2
        seq_len = 10
        latent_code = torch.randn(batch_size, feature_dim, seq_len)
        reference_features = torch.randn(batch_size, feature_dim, 1)
        reference_attention_layer = ReferenceAttentionLayer(feature_dim)
        output = reference_attention_layer(latent_code, reference_features)
        self.assertEqual(output.shape, (batch_size, feature_dim, seq_len))

class TestBackboneNetwork(unittest.TestCase):
    def test_output_shape(self):
        feature_dim = 512
        num_layers = 3
        batch_size = 2
        seq_len = 10
        latent_code = torch.randn(batch_size, feature_dim, seq_len)
        audio_features = torch.randn(batch_size, feature_dim, seq_len)
        ref_image = torch.randn(batch_size, 3, 256, 256)
        reference_net = ReferenceNet()
        audio_attention_layers = AudioAttentionLayers(feature_dim, num_layers)
        temporal_module = TemporalModule()
        backbone_network = BackboneNetwork(feature_dim, num_layers, reference_net, audio_attention_layers, temporal_module)
        output = backbone_network(latent_code, audio_features, ref_image)
        self.assertEqual(output.shape, (batch_size, feature_dim, seq_len))

class TestFramesEncodingVAE(unittest.TestCase):
    def test_output_shape(self):
        latent_dim = 256
        img_size = 256
        batch_size = 2
        num_frames = 4
        reference_image = torch.randn(batch_size, 3, img_size, img_size)
        motion_frames = torch.randn(batch_size, num_frames, 3, img_size, img_size)
        speed_value = torch.randn(batch_size, 1)
        frames_encoding_vae = FramesEncodingVAE(latent_dim, img_size, None)
        reconstructed_frames = frames_encoding_vae(reference_image, motion_frames, speed_value)
        self.assertEqual(reconstructed_frames.shape, (batch_size, num_frames + 1, 3, img_size, img_size))

class TestEMOModel(unittest.TestCase):
    def test_output_shape(self):
        latent_dim = 256
        img_size = 256
        batch_size = 2
        num_frames = 4
        num_timesteps = 100
        noisy_latents = torch.randn(batch_size, num_frames, latent_dim, img_size // 8, img_size // 8)
        timesteps = torch.randint(0, num_timesteps, (batch_size,))
        ref_image = torch.randn(batch_size, 3, img_size, img_size)
        motion_frames = torch.randn(batch_size, num_frames, 3, img_size, img_size)
        audio_features = torch.randn(batch_size, num_frames, 512)
        head_rotation_speeds = torch.randn(batch_size, num_frames)
        vae = VAE()
        image_encoder = ImageEncoder()
        config = {}  # Provide the necessary configuration
        emo_model = EMOModel(vae, image_encoder, config)
        output = emo_model(noisy_latents, timesteps, ref_image, motion_frames, audio_features, head_rotation_speeds)
        self.assertEqual(output.shape, (batch_size, num_frames, latent_dim, img_size // 8, img_size // 8))


# class TestSpeedEncoder(unittest.TestCase):
#     def setUp(self):
#         # Initialize SpeedEncoder with example parameters
#         num_speed_buckets = 9  # Example parameter, adjust as necessary
#         speed_embedding_dim = 128  # Example parameter, adjust as necessary
#         self.speed_encoder = SpeedEncoder(num_speed_buckets, speed_embedding_dim)
    
#     def test_speed_encoder_initialization(self):
#         # Test whether SpeedEncoder initializes correctly with given parameters
#         self.assertIsInstance(self.speed_encoder, SpeedEncoder, "SpeedEncoder did not initialize correctly.")

#     def test_speed_encoder_output(self):
#         # Assuming SpeedEncoder has a method to encode or process inputs, we test it here
#         # Example input, adjust according to the actual method signature
#         input_speed = 5  # Example speed value, adjust as necessary
#         output = self.speed_encoder.encode_speed(input_speed)
        
#         # Example assertion, adjust based on expected output shape or properties
#         self.assertEqual(output.shape, (speed_embedding_dim), "Output shape of SpeedEncoder does not match expected.")




if __name__ == '__main__':
    unittest.main()
