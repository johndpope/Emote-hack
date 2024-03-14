import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from decord import AudioReader
from Net import EMOModel
import decord

# Load the trained EMO model
model_path = 'emo_model_stage3.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the EMOModel
emo_model = EMOModel(
    vae=None,
    image_encoder=None,
    config={
        "feature_dim": 512,
        "num_layers": 4,
        "audio_feature_dim": 128,
        "audio_num_layers": 2,
        "num_speed_buckets": 5,
        "speed_embedding_dim": 64,
        "temporal_module": "conv"
    }
).to(device)

# Load the trained weights
emo_model.load_state_dict(torch.load(model_path, map_location=device))
emo_model.eval()

# Define the necessary transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the reference image
reference_image_path = 'path/to/reference/image.jpg'
reference_image = Image.open(reference_image_path).convert('RGB')
reference_image = transform(reference_image).unsqueeze(0).to(device)

# Load the audio frames
audio_path = 'path/to/audio/file.mp3'
audio_reader = AudioReader(audio_path, ctx=decord.cpu(), sample_rate=16000, mono=True)
audio_frames = audio_reader[:]

# Specify the target head rotation speed - WHAT ??? TODO - fix this
target_speed = 0.5

# Generate the video frames
output_video_path = 'video.mp4'
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))

with torch.no_grad():
    for i in range(len(audio_frames)):
        audio_frame = audio_frames[i].unsqueeze(0).to(device)
        
        # Perform inference
        generated_frame = emo_model(reference_image, audio_frame, target_speed)
        
        # Convert the generated frame tensor to an array and adjust color channels
        generated_frame = generated_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        generated_frame = (generated_frame * 0.5 + 0.5) * 255
        generated_frame = cv2.cvtColor(generated_frame.astype('uint8'), cv2.COLOR_RGB2BGR)
        
        video_writer.write(generated_frame)

video_writer.release()
