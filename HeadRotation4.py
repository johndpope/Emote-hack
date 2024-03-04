import cv2
import mediapipe as mp
import numpy as np
import math
from math import cos, sin, pi
# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define camera matrix based on image dimensions.
def get_camera_matrix(width, height, focal_length):
    return np.array([[focal_length, 0, width / 2],
                     [0, focal_length, height / 2],
                     [0, 0, 1]], dtype=np.float64)

# Function to calculate Euler angles from a rotation matrix.
def rotation_matrix_to_euler_angles(rmat):
    sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rmat[2,1], rmat[2,2])
        y = math.atan2(-rmat[2,0], sy)
        z = math.atan2(rmat[1,0], rmat[0,0])
    else:
        x = math.atan2(-rmat[1,2], rmat[1,1])
        y = math.atan2(-rmat[2,0], sy)
        z = 0
    return np.array([x, y, z])

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img

# Main function to estimate head pose.
def estimate_head_pose(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    img_h, img_w, _ = image.shape
    
    # Landmarks to use for pose estimation.
    landmarks_idx = [1, 33, 263, 61, 291, 199]
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get 2D and 3D landmarks for pose estimation.
        image_points = np.array([(face_landmarks.landmark[idx].x * img_w,
                                  face_landmarks.landmark[idx].y * img_h) for idx in landmarks_idx], dtype=np.float64)
        
        # The 3D points are hardcoded based on the average face model.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Camera internals.
        focal_length = img_w
        cam_matrix = get_camera_matrix(img_w, img_h, focal_length)
        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
        
        # SolvePnP to get rotation vector.
        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if success:
            # Convert rotation vector to matrix.
            rmat, _ = cv2.Rodrigues(rot_vec)
            
            # Calculate Euler angles from rotation matrix.
            euler_angles = rotation_matrix_to_euler_angles(rmat)
            
            # Convert angles to degrees.
            euler_angles_deg = euler_angles * (180.0 / np.pi)
            yaw, pitch, roll = euler_angles_deg
            
            # Draw the pose on the image if debugging is enabled.
            if True:
                draw_axis(image, yaw, pitch, roll)
                debug_image_path = image_path.replace('.jpg', '_debug.jpg')  # Modify as needed
                cv2.imwrite(debug_image_path, image)
                print(f'Debug image saved to {debug_image_path}')
            
            return euler_angles_deg
        else:
            return None
    

# Usage example.
image_path =  'frame_0094.jpg' 
head_pose = estimate_head_pose(image_path)

roll, pitch, yaw = head_pose
print(f'Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}')
