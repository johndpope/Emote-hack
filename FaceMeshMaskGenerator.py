import cv2
import numpy as np
import torch
import mediapipe as mp

class FaceMeshMaskGenerator:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

    #  bbox for face
    def detect_face_region(image):
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



# Example usage
# image = cv2.imread('path/to/image.jpg')
# face = FaceMeshMaskGenerator()
# face_region = face.detect_face_region(image)
# latent_dim = 512
# num_latents = 5  # For example, to create 5 different noisy latents
# noisy_latents = face.generate_noisy_latents(face_region, latent_dim, num_latents)