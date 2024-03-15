
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import numpy as np
import torchaudio
from dataset import AudioVisualDataset
import torch
from torchvision import models, transforms
from PIL import Image
import logging
import os



class Wav2VecFeatureExtractor:
    def __init__(self, model_name='facebook/wav2vec2-base-960h', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)

    def extract_features(self, waveform, sample_rate=16000):
        """
        Extract audio features from a waveform using Wav2Vec 2.0.

        Args:
            waveform (Tensor): The waveform of the audio.
            sample_rate (int): The sample rate of the waveform.

        Returns:
            torch.Tensor: Features extracted from the audio.
        """

        # Ensure waveform is a 2D tensor (channel, time)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension if it's not present
        elif waveform.ndim == 3:
            # If there is a batch dimension, we squeeze it out, because the model expects 2D input
            # This is true for a batch size of 1. If the batch size is greater, further handling is needed.
            waveform = waveform.squeeze(0)

        # Process the audio to extract features
        input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values


        # Reshape input_values to remove the extraneous dimension if present
        if input_values.ndim == 3 and input_values.size(0) == 1:
            input_values = input_values.squeeze(0)

        # Check the shape of input_values to be sure it's 2D now
        assert input_values.ndim == 2, "Input_values should be 2D (batch, sequence_length) after squeezing"

        input_values = input_values.to(self.device)

        # Pass the input_values to the model
        with torch.no_grad():
            features = self.model(input_values).last_hidden_state

        # Reduce the dimensions if necessary, usually you get (batch, seq_length, features)
        # You might want to average the sequence length dimension or handle it appropriately
        features = features.mean(dim=1)  # Example of reducing the sequence length dimension by averaging
        return features




# Implement the CrossAttentionLayer and AudioAttentionLayers
class CrossAttentionLayer(nn.Module):
    def __init__(self, feature_dim, device):
        super(CrossAttentionLayer, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim).to(device)
        self.key = nn.Linear(feature_dim, feature_dim).to(device)
        self.value = nn.Linear(feature_dim, feature_dim).to(device)
        # Initialize scale on the correct device
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim])).to(device)

    def forward(self, latent_code, audio_features, return_attention_weights=True):
        query = self.query(latent_code)
        key = self.key(audio_features)
        value = self.value(audio_features)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value)

        if return_attention_weights:
            return attention_output, attention_probs  # Return both output and attention weights
        return attention_output, None  # If not returning weights, return None in place of weights



class AudioAttentionLayers(nn.Module):
    def __init__(self, feature_dim, num_layers, device):
        super(AudioAttentionLayers, self).__init__()
        self.layers = nn.ModuleList([CrossAttentionLayer(feature_dim, device) for _ in range(num_layers)])

    def forward(self, latent_code, audio_features, return_attention_weights=False):
            attention_weights = []
            for layer in self.layers:
                output, weights = layer(latent_code, audio_features, return_attention_weights=True)
                latent_code = output + latent_code
                if return_attention_weights:
                    attention_weights.append(weights)

            if return_attention_weights:
                # Stack the weights from each layer to form a tensor
                return latent_code, torch.stack(attention_weights)
            return latent_code
    
class FeatureTransformLayer(nn.Module):
    def __init__(self, input_dim, output_dim,device):
        super(FeatureTransformLayer, self).__init__()
        self.transform = nn.Linear(input_dim, output_dim).to(device)
    
    def forward(self, features):
        # print("FeatureTransformLayer input device:", features.device)  # Log device
        return self.transform(features)
