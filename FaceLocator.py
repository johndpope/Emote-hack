import torch
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torch
import torch.optim as optim
import mediapipe as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class FaceLocator(nn.Module):
    def __init__(self):
        super(FaceLocator, self).__init__()
        # Define lightweight convolutional layers to encode the bounding box
        # Assuming input images are (C, H, W) = (3, 256, 256) for example
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, images):
        # images shape is (B, C, H, W)
        B, C, H, W = images.shape
        # Example forward pass through the lightweight convolutional layers
        x = F.relu(self.conv1(images))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # Final shape (B, 64, H/8, W/8)
        
        # We assume the final layer's output is an encoded representation 
        # of the facial bounding box which we can transform into a mask.
        # Generate a mask from the output - this can be a learned transformation
        # or a fixed operation depending on the model's design.
        # For the sake of example, we'll threshold the activations to create a mask.
        masks = x > x.mean(dim=[2, 3], keepdim=True)  # Simple thresholding for mask
        masks = F.interpolate(masks.float(), size=(H, W), mode='nearest')  # Upscale to original size
        
        return masks


#  bbox for face
def detect_face_region(image):
    assert isinstance(image, np.ndarray), "Input must be a numpy array"
    assert len(image.shape) == 3 and image.shape[2] == 3, "Image must be in HWC format with 3 channels"

    # Initialize MediaPipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(image_rgb)

    # Check if exactly one face is detected
    if not results.detections or len(results.detections) != 1:
        raise ValueError("Exactly one face must be detected in the image.")

    # Extract the bounding box of the face
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    h, w, _ = image.shape
    x, y, w, h = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
    face_region = image[y:y+h, x:x+w]
    return face_region
    

def generate_noisy_latents(face_region, latent_dim, num_latents=1):
    # Resize face region to match latent dimension
    resized_face = cv2.resize(face_region, (latent_dim, latent_dim))

    # Normalize face region to range [0, 1]
    normalized_face = resized_face.astype(np.float32) / 255.0

    # Initialize array to hold all noisy latents
    noisy_latents = []

    # Generate multiple noisy latents
    for _ in range(num_latents):
        # Generate random noise
        noise = np.random.normal(0, 1, (latent_dim, latent_dim, 3))

        # Add noise to the normalized face region
        noisy_latent = normalized_face + noise

        # Clip values to range [0, 1]
        noisy_latent = np.clip(noisy_latent, 0, 1)

        # Append the noisy latent to the array
        noisy_latents.append(noisy_latent)

    return np.array(noisy_latents)



class FaceMaskGenerator:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

    def generate_mask(self, image):

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a blank mask with the same dimensions as the image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Detect faces
        with self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            detection_results = face_detection.process(image_rgb)

        # If faces are detected, find the face landmarks.
        if detection_results.detections:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5) as face_mesh:
                
                # Apply the face mesh model.
                mesh_results = face_mesh.process(image_rgb)
                
                if mesh_results.multi_face_landmarks:
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        # Draw the face mesh on the mask
                        for landmark in face_landmarks.landmark:
                            x = min(int(landmark.x * image.shape[1]), image.shape[1] - 1)
                            y = min(int(landmark.y * image.shape[0]), image.shape[0] - 1)
                            mask[y, x] = 255

        # Optional: Expand the mask to cover a larger area if needed
        # For example, you can use cv2.dilate() to expand the mask area

        # Convert the mask to a PyTorch tensor
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        return mask_tensor

# Define your dataset and data loader
transform = transforms.Compose([
    # Add any required transformations here
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root='path_to_training_images', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Instantiate the model and optimizer
model = FaceLocator()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Instantiate the face mask generator
face_mask_generator = FaceMaskGenerator()

# Define the training loop
def train_model(model, data_loader, face_mask_generator, optimizer, num_epochs):
    assert isinstance(num_epochs, int) and num_epochs > 0, "Number of epochs must be a positive integer"
    
    model.train()
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            assert images.ndim == 4, "Images must be a 4D tensor"
            assert images.size(1) == 3, "Images must have 3 channels"
            optimizer.zero_grad()

       
            # Generate face masks for each image in the batch
            face_masks = []
            for img in images.numpy():
                img = (img * 255).astype(np.uint8).transpose(1, 2, 0)  # Convert to uint8 and HWC format for cv2
                mask = face_mask_generator.generate_mask(img)
                face_masks.append(mask)

            # Convert list of masks to a tensor
            face_masks_tensor = torch.stack(face_masks)

            # Forward pass: compute the output of the model using images and masks
            outputs = model(images, face_masks_tensor)


            # Ensure the mask is the same shape as the model's output
            face_masks_tensor = F.interpolate(face_masks_tensor.unsqueeze(1), size=outputs.shape[2:], mode='nearest').squeeze(1)
            assert outputs.shape == face_masks_tensor.shape, "Output and face masks tensor must have the same shape"

             # Compute the loss
            loss = criterion(outputs, face_masks_tensor)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}')

# Call the training loop
train_model(model, data_loader, face_mask_generator, optimizer, num_epochs=25)
