import torchaudio
from moviepy.editor import VideoFileClip
import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import torch.nn as nn

# This is a dummy example of a neural network module that might take the concatenated frame features
class AudioFeatureModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(AudioFeatureModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)
    

# Load the processor and model from HuggingFace Transformers
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def get_nearby_frames_features(A, f, m, n):
    """
    Concatenate feature vectors from nearby frames.
    
    Args:
        A (torch.Tensor): Tensor of shape (num_frames, feature_dim) containing the feature vectors.
        f (int): The current frame index.
        m (int): The number of frames before the current frame to include.
        n (int): The number of frames after the current frame to include.
    
    Returns:
        torch.Tensor: The concatenated feature vector for frame f.
    """
    # Calculate start and end frame indices
    start_frame = max(f - m, 0)
    end_frame = min(f + n + 1, A.size(0))
    
    # Extract the nearby frames
    nearby_frames = A[start_frame:end_frame]
    
    # Determine if we need to pad at the beginning
    pad_front = max(0, m - f)
    
    # Determine if we need to pad at the end
    pad_back = max(0, (f + n + 1) - A.size(0))
    
    # Apply padding if necessary
    if pad_front > 0 or pad_back > 0:
        padding = (pad_front, pad_back)
        nearby_frames = torch.nn.functional.pad(nearby_frames, pad=padding, mode='constant', value=0)

    # Flatten the nearby frames into a single feature vector
    return nearby_frames.flatten()


# Set device for torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained Wav2Vec model
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(device)
model.eval()


import numpy as np





def extract_features_from_mp4(video_path, model_name='facebook/wav2vec2-base-960h'):
    """
    Extract audio features from an MP4 file using Wav2Vec 2.0.

    Args:
    video_path (str): Path to the MP4 video file.
    model_name (str): Model name or path.

    Returns:
    torch.Tensor: Features extracted from the audio.
    """
    # Load pretrained Wav2Vec 2.0 tokenizer and model from Hugging Face Hub
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)

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
    if sample_rate != processor.feature_extractor.sampling_rate:
        waveform = librosa.resample(np.float32(waveform), orig_sr=sample_rate, target_sr=processor.feature_extractor.sampling_rate)
        sample_rate = processor.feature_extractor.sampling_rate
    # # Ensure waveform is a 1D array for a single-channel audio
    # if waveform.ndim > 1:
    #     waveform = waveform.squeeze()
    # Ensure waveform is a 1D array for a single-channel audio
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)  # Taking mean across channels for simplicity

    print("waveform.ndim: ", waveform.ndim)
    # # Check and adjust dimensions
    # # Ensure waveform is 2D (shape: [channels, sequence_length]) or 3D (shape: [batch_size, channels, sequence_length])
    # if waveform.ndim == 4:
    #     waveform = waveform.squeeze()  # Removes the additional dimension
    # elif waveform.ndim == 2:
    #     waveform = waveform.unsqueeze(0)  # Adds a batch dimension
        

    # Check the current length of the waveform
    current_length = waveform.shape[0]
    print("current_length: ", current_length)
    # Define the desired length (for example, 16000 if you want 1 second of audio at 16kHz)
    desired_length = 16000

    # Calculate the padding length needed
    padding_length = max(desired_length - current_length, 0)
    print("padding_length: ", padding_length)
    # Pad the waveform if necessary
    if padding_length > 0:
        waveform = np.pad(waveform, (0, padding_length), mode='constant')

    # Process the audio to extract features
    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
    input_values = input_values.to('cuda' if torch.cuda.is_available() else 'cpu')


    # Pass the input_values to the model
    with torch.no_grad():
        hidden_states = model(input_values).last_hidden_state
        
    return hidden_states



# Example usage:
video_path = 'M2Ohb0FAaJU_1.mp4'
A = extract_features_from_mp4(video_path)
print(A.shape)

# Ensure A is squeezed to remove the batch dimension if it's size 1 (otherwise -  Input tensor A should be 2D but has shape torch.Size([1, 629, 768]))
A = A.squeeze(0) if A.size(0) == 1 else A

# Assuming A is a 2D tensor of shape [num_frames, feature_dim]
num_frames, feature_dim = A.shape[0], A.shape[1]
m, n = 2, 2  # Number of frames before and after
input_size = feature_dim * (m + n + 1)  # Update input size accordingly
output_size = 768  # Example output size (you may need to change this based on your task)


def get_nearby_frames_features(A, f, m, n, feature_dim):
    """
    Concatenate feature vectors from nearby frames.
    
    Args:
        A (torch.Tensor): Tensor of shape (num_frames, feature_dim) containing the feature vectors.
        f (int): The current frame index.
        m (int): The number of frames before the current frame to include.
        n (int): The number of frames after the current frame to include.
        feature_dim (int): The dimension of the feature vectors.
    
    Returns:
        torch.Tensor: The concatenated feature vector for frame f.
    """
    assert A.dim() == 2, f"Input tensor A should be 2D but has shape {A.shape}"
    assert A.size(1) == feature_dim, f"Feature dimension of A does not match feature_dim, got: {A.size(1)}, expected: {feature_dim}"

    num_frames = A.size(0)
    start_frame = max(f - m, 0)
    end_frame = min(f + n + 1, num_frames)
    nearby_frames_features = A[start_frame:end_frame].clone()

    # Assert that nearby_frames_features is 2D
    assert nearby_frames_features.dim() == 2, f"nearby_frames_features should be 2D but has shape {nearby_frames_features.shape}"

    # Add padding if necessary
    if f - m < 0:
        front_padding = torch.zeros((m - f, feature_dim), device=A.device)
        nearby_frames_features = torch.cat((front_padding, nearby_frames_features), dim=0)
    if f + n + 1 > num_frames:
        end_padding = torch.zeros((f + n + 1 - num_frames, feature_dim), device=A.device)
        nearby_frames_features = torch.cat((nearby_frames_features, end_padding), dim=0)

    # Assert that the final tensor is 2D before flattening
    assert nearby_frames_features.dim() == 2, f"nearby_frames_features should be 2D but has shape {nearby_frames_features.shape} before flattening"

    return nearby_frames_features.view(-1)  # Flatten the features


# Create the model
model = AudioFeatureModel(input_size, output_size).to(device)

# Process all frames and get features
all_features = []
for f in range(num_frames):
    # Get features for the current frame and its neighbors
    nearby_frames_features = get_nearby_frames_features(A, f, m, n, feature_dim)
    print(nearby_frames_features.shape)  # This should print the shape of the concatenated features

    # Convert to PyTorch tensor and add batch dimension
    frame_features = nearby_frames_features.unsqueeze(0).to(device)
    
    # Pass through the model
    output = model(frame_features)
    
    # Store the output
    all_features.append(output)

# Convert list of tensors to a single tensor
all_features = torch.cat(all_features, dim=0)
print("all_features.shape: ", all_features.shape)  # Output shape: (num_frames, output_size)


