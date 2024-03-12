import torch
import cv2
import numpy as np
import mediapipe as mp

import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image

os.environ["OPENCV_LOG_LEVEL"]="FATAL"

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
        # Initialize FaceDetection once here
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    def __del__(self):
        self.face_detection.close()
        self.face_mesh.close()

  

    def generate_face_region_mask_pil_image(self, frame_image):
        # Convert from PIL Image to NumPy array in BGR format
        frame_np = np.array(frame_image.convert('RGB'))  # Ensure the image is in RGB
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        height, width, _ = frame_bgr.shape

        # Create a blank mask with the same dimensions as the frame
        mask = np.zeros((height, width), dtype=np.uint8)

        # Detect faces
        detection_results = self.face_detection.process(frame_bgr)

        # Assert that detection_results is not None
        assert detection_results is not None, "The face detection results must not be None."

        # Create a debug image by copying the BGR frame TODO - these need to have more padding.
        # sometimes videos are too dark to detect faces...
        if False:
            debug_image = frame_bgr.copy()

        # If faces are detected
            if detection_results.detections:
                for detection in detection_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    xmin = int(bboxC.xmin * width)
                    ymin = int(bboxC.ymin * height)
                    bbox_width = int(bboxC.width * width)
                    bbox_height = int(bboxC.height * height)

                    # Draw a rectangle on the debug image for each detection
                    cv2.rectangle(debug_image, (xmin, ymin), (xmin + bbox_width, ymin + bbox_height), (0, 255, 0), 2)

        # # Save the input frame as a debug image (before drawing rectangles)
            cv2.imwrite('input_frame.png', frame_bgr)

        # # Save the debug image with rectangles around detected faces
            cv2.imwrite('debug_image.png', debug_image)

        # If faces are detected, find the face landmarks
        if detection_results.detections:
            # Apply the face mesh model
            mesh_results = self.face_mesh.process(frame_bgr)

            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = min(int(landmark.x * width), width - 1)
                        y = min(int(landmark.y * height), height - 1)
                        mask[y, x] = 255

        return mask
