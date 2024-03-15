import os
from PIL import Image
from torch.utils.data import Dataset
import torchaudio
import torchvision

class AudioVisualDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.audio_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.wav')])
        self.image_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.png')])
        
        assert len(self.audio_files) == len(self.image_files), "Number of audio files and image files should match"
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.dataset_dir, self.audio_files[idx])
        image_path = os.path.join(self.dataset_dir, self.image_files[idx])
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        assert waveform is not None, "Failed to load waveform"
        # Handling stereo audio by averaging the channels to convert to mono
        # This is just one approach; depending on your needs you might handle it differently
        if waveform.ndim == 2 and waveform.size(0) == 2:  # Check if the audio is stereo
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono by averaging the channels

        # Check if waveform has an unexpected batch dimension and remove it
        if waveform.ndim == 3:
            assert waveform.size(0) == 1, "Batch size should be 1"
            waveform = waveform.squeeze(0)  # Remove the batch dimension

        # Resample audio if necessary
        target_sample_rate = 16000  # Set your desired sample rate
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        assert waveform.ndim == 2, "Waveform should have 2 dimensions after processing (channels, time)"


        # Load image
        image = Image.open(image_path).convert('RGB')
        assert image is not None, "Failed to load image"

        # Convert image to tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
        ])
        image = transform(image)
        assert image is not None, "Failed to transform image"

        return {
            'audio': waveform,
            'image': image
        }