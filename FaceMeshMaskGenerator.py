import cv2
import numpy as np
import torch
import mediapipe as mp

class FaceMeshMaskGenerator:
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
