import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

class FramesEncoder:
    def __init__(self, frame_size=(64, 64), use_feature_extractor=False):
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.use_feature_extractor = use_feature_extractor
        if self.use_feature_extractor:
            self.feature_extractor = resnet50(pretrained=True)
            self.feature_extractor.eval()  # Set to evaluation mode

    def preprocess_frame(self, frame):
        # Apply transformations
        frame = self.transform(frame)
        return frame

    def extract_features(self, frame):
        # Assuming frame is already preprocessed
        with torch.no_grad():
            features = self.feature_extractor(frame.unsqueeze(0))  # Add batch dimension
        return features.squeeze(0)  # Remove batch dimension

    def load_frames_from_folder(self, folder_path):
        frames = []
        # Sort files to ensure they are processed in order
        file_list = sorted(os.listdir(folder_path))
        for file_name in file_list:
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                frame_path = os.path.join(folder_path, file_name)
                frame = Image.open(frame_path)
                frames.append(frame)
        return frames

    def encode(self, folder_path):
        frames = self.load_frames_from_folder(folder_path)
        # Preprocess and possibly extract features from each frame
        processed_frames = [self.preprocess_frame(frame) for frame in frames]
        
        if self.use_feature_extractor:
            processed_frames = [self.extract_features(frame) for frame in processed_frames]

        # Stack frames into a single tensor
        frame_tensor = torch.stack(processed_frames)
        return frame_tensor

# Example usage
# frames_encoder = FramesEncoder(use_feature_extractor=True)
# folder_path = 'path_to_extracted_frames_folder'  # Replace with the actual path to your frames folder
# video_frame_tensor = frames_encoder.encode(folder_path)

# Now, video_frame_tensor can be used as input to the EMO model or any other application
