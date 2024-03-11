import cv2
import numpy as np
import mediapipe as mp

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

def generate_noisy_latent(face_region, latent_dim):
    # Resize face region to match latent dimension
    resized_face = cv2.resize(face_region, (latent_dim, latent_dim))

    # Normalize face region to range [0, 1]
    normalized_face = resized_face.astype(np.float32) / 255.0

    # Generate random noise
    noise = np.random.normal(0, 1, (latent_dim, latent_dim, 3))

    # Add noise to the normalized face region
    noisy_latent = normalized_face + noise

    # Clip values to range [0, 1]
    noisy_latent = np.clip(noisy_latent, 0, 1)

    return noisy_latent

# Example usage
image = cv2.imread('path/to/image.jpg')
face_region = detect_face_region(image)
latent_dim = 512
noisy_latent = generate_noisy_latent(face_region, latent_dim)