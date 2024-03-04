import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define the landmarks that represent the head pose.
HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

def get_head_pose(image: np.array):
    """
    Given an image, estimate the head pose (roll, pitch, yaw angles).

    Args:
        image: Image to estimate head pose.

    Returns:
        tuple: Roll, Pitch, Yaw angles if face landmarks are detected, otherwise None.
    """
    # Convert the image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect face landmarks.
    results = face_mesh.process(image_rgb)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in HEAD_POSE_LANDMARKS:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            # Convert to numpy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix
            focal_length = img_w  # Assuming fx = fy
            cam_matrix = np.array(
                [[focal_length, 0, img_w / 2],
                 [0, focal_length, img_h / 2],
                 [0, 0, 1]]
            )

            # Distortion matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP to get rotation vector
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )

            if success:
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get Euler angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                roll = angles[0] * 180 / np.pi
                pitch = angles[1] * 180 / np.pi
                yaw = angles[2] * 180 / np.pi

                return roll, pitch, yaw

    return None

# Usage example
image_path = 'frame.png'  # Replace with your image path
image = cv2.imread(image_path)

if image is not None:
    head_pose = get_head_pose(image)
    if head_pose:
        roll, pitch, yaw = head_pose
        print(f'Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}')
    else:
        print('No face detected or the face landmarks are not sufficient for pose estimation.')
