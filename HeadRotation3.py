import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

factor = 360*10  # Scaling factor
def calculate_head_rotation(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    img_h, img_w, _ = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x, y = lm.x * img_w, lm.y * img_h
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z * img_w])  # lm.z is scaled by img_w for the 3D coordinates

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix based on image dimensions
            focal_length = img_w  # Approximation for focal length
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)  # Assuming no lens distortion

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            if success:
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                roll, pitch, yaw = angles  # Angles are in radians

                # Convert angles to degrees
                roll, pitch, yaw = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

                return roll, pitch, yaw
    return None


# Usage example
image_path = 'frame_0094.jpg'  # Replace with the path to your image
head_rotation = calculate_head_rotation(image_path)
if head_rotation:
    roll, pitch, yaw = head_rotation
    print(f'Roll: {roll:.4f}, Pitch: {pitch:.4f}, Yaw: {yaw:.4f}')
else:
    print('Head rotation could not be calculated.')
