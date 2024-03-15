import os
import torchvision
import torch

# pip3 install --U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
from diffusers import StableDiffusionPipeline
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

# Set up the AudioGen model
model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=1)  # Generate 1-second audio samples

# Set up the Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_auth_token=True)
pipe = pipe.to(device)

# Define the audio descriptions and corresponding image descriptions
descriptions = ['beep', 'tick', 'buzz']
image_descriptions = ['a circle on a white background', 'a square on a white background', 'a triangle on a white background']

# Set up the output directory
output_dir = 'synthetic_dataset'
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic audio-visual pairs
num_samples = 10  # Number of samples to generate for each description pair
for desc, img_desc in zip(descriptions, image_descriptions):
    for i in range(num_samples):
        # Generate audio
        wav = model.generate([desc])
        audio_filename = f"{desc}_{i}.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        audio_write(audio_path, wav[0].cpu(), model.sample_rate, strategy="loudness")

        # Generate corresponding image using Stable Diffusion
        image_filename = f"{desc}_{i}.png"
        image_path = os.path.join(output_dir, image_filename)
        with torch.autocast("cuda"):
            image = pipe(img_desc).images[0]
        image.save(image_path)

        print(f"Generated {audio_filename} and {image_filename}")

print("Synthetic dataset generation completed.")