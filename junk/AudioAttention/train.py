
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
from model import Wav2VecFeatureExtractor,FeatureTransformLayer,AudioAttentionLayers

# Load a pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to inference mode


# run this like 
#  python ./junk/AudioAttention/train.py


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Checkpoint directory
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate a transformation layer for visual features
visual_feature_transform = FeatureTransformLayer(input_dim=2048, output_dim=768,device=device)
# visual_feature_transform = visual_feature_transform.to(device)  # Move to GPU if necessary
# 

# Instantiate the Wav2VecFeatureExtractor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Instantiate the AudioAttentionLayers
audio_attention = AudioAttentionLayers(feature_dim=768, num_layers=3, device=device)



# Pre-load and set the ResNet-50 model to inference mode
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()
resnet_model.to(device)  # Assuming using CUDA

# Remove the final fully connected layer to extract features instead of class predictions
resnet_feature_extractor = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
resnet_feature_extractor.to(device)

# Instantiate the dataset and data loader
dataset = AudioVisualDataset('./junk/AudioAttention/synthetic_dataset')  # launching with vscode / settings
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Set up the optimizer and loss function
optimizer = optim.Adam(audio_attention.parameters(), lr=0.001)
criterion = nn.MSELoss()


# Instantiate the Wav2VecFeatureExtractor
audio_feature_extractor = Wav2VecFeatureExtractor('facebook/wav2vec2-base-960h',device)

# Training loop
num_epochs = 10



# Initialize metric accumulators
total_loss = 0.0
total_samples = 0
log_interval = 10
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(data_loader):
         # Check that 'audio' and 'image' keys exist in the batch
        assert 'audio' in batch and 'image' in batch, "Batch must contain 'audio' and 'image' keys"

         # Assuming 'audio' and 'image' are correctly loaded and 'audio' is waveform
        waveform = batch['audio']  # Assuming sample_rate is 16000 for all waveforms
        waveform = waveform.to(device)
        # Assuming 'waveform' is a 3D tensor from the DataLoader
        for i in range(waveform.size(0)):  # Iterate over the batch
            single_waveform = waveform[i]  # Extract a 2D tensor (channels, time)
            # Process single_waveform as needed
        
            # Extract audio features
            audio_features = audio_feature_extractor.extract_features(single_waveform)

            # Assuming batch["image"] is a tensor of shape (batch_size, C, H, W)
            image = batch["image"].to(device)


            # Ensure the waveform and image tensors are not empty and have expected dimensions
            assert waveform.ndim == 3, "Waveform tensor should have 3 dimensions (batch, channels, length)"
            assert image.ndim == 4, "Image tensor should have 4 dimensions (batch, channels, height, width)"

            # Preprocess and extract features for the whole batch        with torch.no_grad():
            visual_features = resnet_feature_extractor(image)


            # Depending on the resnet_feature_extractor output, you might need to adjust its dimensions
            # If the feature extractor outputs a tensor of shape (batch_size, features, 1, 1),
            # you should remove the last two dimensions:
            visual_features = torch.flatten(visual_features, start_dim=1)

            # Move visual features to the same device as the transformation layer
            visual_features = visual_features.to(device)
             # Transform visual features to match the attention layer input dimensionality
            visual_features_transformed = visual_feature_transform(visual_features)

            # Assert that features have the correct dimensions
            # print("🎉  audio_features:",audio_features.ndim)
            assert audio_features.ndim == 2, "Audio features should be 2D (batch, features)"
            assert visual_features_transformed.ndim == 2, "Visual features should be 2D (batch, features)"

            optimizer.zero_grad()
            attended_features,attention_weights =  audio_attention(
            visual_features_transformed, 
            audio_features, 
            return_attention_weights=True
        )
            # Assert the attended features have the correct shape
            assert attended_features.ndim == 2, "Attended features should be 2D (batch, features)"


            loss = criterion(attended_features, visual_features_transformed)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * waveform.size(0)
            total_samples += waveform.size(0)
                   # Accumulate loss for the epoch
            epoch_loss += loss.item()

            # Logging within the batch loop if needed
            if batch_idx % log_interval == 0:  # Assuming you define log_interval
                logger.info(f'Epoch: {epoch+1} [{batch_idx * len(waveform)}/{len(data_loader.dataset)} '
                            f'({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluation and visualization
# Calculate the average loss across all validation samples
average_loss = total_loss / total_samples



# Comparison with baselines
# Here, you should load the performance metrics of baseline models or previous studies
# For demonstration, let's assume you have these as constants
baseline_loss = 0.05  # hypothetical value

 # Average loss for the epoch
epoch_loss /= len(data_loader.dataset)
logger.info(f'Epoch {epoch+1} finished, average loss: {epoch_loss:.6f}')

# Saving checkpoints after the epoch
checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
torch.save({
    'epoch': epoch+1,
    'model_state_dict': audio_attention.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
}, checkpoint_path)

logger.info(f'Checkpoint saved to {checkpoint_path}')

# Rest of your code...

# Instead of printing, use logger to log the final results
logger.info(f'Average validation loss: {average_loss:.4f}')
logger.info(f'Comparison with Baseline: Baseline Loss - {baseline_loss:.4f}, Model Loss - {average_loss:.4f}')