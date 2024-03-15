import torch
from torchvision import models, transforms
from PIL import Image
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from model import Wav2VecFeatureExtractor,FeatureTransformLayer,AudioAttentionLayers
# Ensure your inference device is set correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# invoke using
# python ./junk/AudioAttention/inference.py

# Load the trained model and other necessary modules for inference
def load_model(checkpoint_path, device):
    # Load pre-trained ResNet-50 and Wav2Vec2.0 models for feature extraction
    resnet_model = models.resnet50(pretrained=True)
    resnet_model.eval()
    resnet_model.to(device)
    
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model.eval()
    wav2vec_model.to(device)

    # Load your custom AudioAttention model
    audio_attention = AudioAttentionLayers(feature_dim=768, num_layers=3, device=device)
    audio_attention.to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    audio_attention.load_state_dict(checkpoint['model_state_dict'])

    return resnet_model, wav2vec_model, audio_attention

# Define the preprocessing for image and audio
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Assuming ResNet-50 feature maps are 2048 in depth before the final pooling and classification layers.
visual_feature_transform = FeatureTransformLayer(input_dim=2048, output_dim=768, device=device)

# Inference function
def inference(image_path, audio_path, resnet_model, wav2vec_model, audio_attention, device):
    # Process the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_transforms(image).unsqueeze(0).to(device)


    # Extract features from the image using ResNet
    with torch.no_grad():
        visual_features = resnet_feature_extractor(image_tensor)
        # Flatten the spatial dimensions:
        visual_features = visual_features.view(visual_features.size(0), 2048, -1).mean(-1)
        visual_features = visual_feature_transform(visual_features)

    # Load and process the audio
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)

    # If the waveform tensor has more than 2 dimensions (batch and channels), reduce it to 2 dimensions
    if waveform.ndim > 2:
        waveform = waveform.squeeze(0)  # Remove the batch dimension if it's present

    # Process the audio to extract features
    with torch.no_grad():
        input_values = audio_processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
        # Check the dimension of input_values and remove any unnecessary dimensions
        if input_values.ndim > 2:
            input_values = input_values.squeeze(0)  # Squeeze out the batch dimension if it's 1
        
        # Ensure input_values is 2D (batch, sequence_length)
        assert input_values.ndim == 2, f"Input_values should be 2D, but got {input_values.size()}"
        
        input_values = input_values.to(device)
        audio_features = wav2vec_model(input_values).last_hidden_state
        audio_features = audio_features.mean(dim=1)

    # Use the AudioAttention model to generate attended features and attention weights
    with torch.no_grad():
        attended_features, attention_weights_list = audio_attention(
            visual_features, 
            audio_features, 
            return_attention_weights=True
        )
    
    # Convert list of attention weights to a tensor if necessary
    if isinstance(attention_weights_list, list):
        # Assuming attention_weights_list is a list of tensors with the same shape
        attention_weights_tensor = torch.stack(attention_weights_list)
    else:
        attention_weights_tensor = attention_weights_list  # if it's already a tensor

    # Now you can index attention_weights_tensor as needed
    # Ensure the indexing matches the dimensions of the tensor
    attention_matrix = attention_weights_tensor[0, 0, :].detach().cpu().numpy()

    return attended_features, attention_matrix


# Load the model (make sure to replace 'checkpoint_epoch_10.pt' with your actual checkpoint file)
resnet_model, wav2vec_model, audio_attention = load_model('./checkpoints/checkpoint_epoch_10.pt', device)

# After loading the ResNet model:
resnet_feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-2])
resnet_feature_extractor.to(device)

# Perform inference (replace 'path_to_image.jpg' and 'path_to_audio.wav' with your actual file paths)
attended_features,attention_weights = inference('./junk/AudioAttention/synthetic_dataset/beep_0.png', './junk/AudioAttention/synthetic_dataset/beep_0.wav.wav', resnet_model, wav2vec_model, audio_attention, device)

# Here, 'attended_features' would be the output of your model
# print("Attended features:", attended_features)
# Assuming attention_weights is a list of tensors
if attention_weights:
    # Access the first layer's weights (check the dimensions to be sure)
    first_layer_weights = attention_weights[0]
    
    if first_layer_weights.ndim == 4:
        # Access the first head's attention weights of the first layer
        attention_matrix = first_layer_weights[0, 0, :, :].detach().cpu().numpy()
    else:
        raise ValueError("Unexpected number of dimensions in attention weights")


# Plotting as a heatmap
sns.heatmap(attention_matrix.reshape(-1, 1), cmap='viridis', cbar=True)
plt.title('Attention Weights Over Audio Sequence')
plt.xlabel('Attention Head')
plt.ylabel('Audio Time Steps')
plt.show()