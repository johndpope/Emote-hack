import cv2
import mediapipe as mp
import numpy as np
from face_geometry import PCF, get_metric_landmarks

def calculate_head_rotation(image):

    
    # Initialize MediaPipe Face Mesh.
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Convert image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect face landmarks.
    results = face_mesh.process(image_rgb)

    # Check if face landmarks are found.
    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    # Extract landmarks for PnP.
    img_h, img_w, _ = image.shape
    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]).T
    pcf = PCF(near=1, far=10000, frame_height=img_h, frame_width=img_w, fy=img_w)  # Assuming square pixels.
    metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)

    # Camera matrix and distance matrix for solvePnP.
    focal_length = img_w
    cam_matrix = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Indices of landmarks to use for PnP.
    points_idx = [33, 263, 61, 291, 199]
    model_points = metric_landmarks[0:3, points_idx].T
    image_points = landmarks[0:2, points_idx].T * np.array([img_w, img_h])[None, :]

    # Solve PnP to get rotation vector.
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, cam_matrix, dist_matrix)

    if not success:
        return None

    # Get rotational matrix.
    rmat, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles from rotation matrix.
    yaw = np.arctan2(rmat[1,0], rmat[0,0])
    pitch = np.arctan2(-rmat[2,0], np.sqrt(rmat[2,1]**2 + rmat[2,2]**2))
    roll = np.arctan2(rmat[2,1], rmat[2,2])

    # Convert angles to degrees.
    pitch, yaw, roll = pitch * 180 / np.pi, yaw * 180 / np.pi, roll * 180 / np.pi

    return pitch, yaw, roll

# Usage example
image_path = 'frame_0094.jpg'
image = cv2.imread(image_path)
if image is not None:
    head_rotation = calculate_head_rotation(image)
    if head_rotation:
        pitch, yaw, roll = head_rotation
        print(f'Pitch: {pitch}, Yaw: {yaw}, Roll: {roll}')
    else:
        print
