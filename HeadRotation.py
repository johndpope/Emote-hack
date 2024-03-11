import cv2
import mediapipe as mp
import numpy as np
import torch
from camera import Camera
from video import Video
import math
from math import cos, sin, pi
from decord import VideoReader

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

def calculate_pose( face2d):
        """Calculates head pose from detected facial landmarks using 
        Perspective-n-Point (PnP) pose computation:
        
        https://docs.opencv.org/4.6.0/d5/d1f/calib3d_solvePnP.html
        """
        # print('Computing head pose from tracking data...')
        # for idx, time in enumerate(self.face2d['time']):
        #     # print(time)
        #     self.pose['time'].append(time)
        #     self.pose['frame'].append(self.face2d['frame'][idx])
        #     face2d = self.face2d['key landmark positions'][idx]
        face3d = [[0, -1.126865, 7.475604], # 1
                       [-4.445859, 2.663991, 3.173422], # 33
                       [-2.456206,	-4.342621, 4.283884], # 61
                       [0, -9.403378, 4.264492], # 199
                       [4.445859, 2.663991, 3.173422], # 263
                       [2.456206, -4.342621, 4.283884]] # 291
        face2d = np.array(face2d, dtype=np.float64)
        face3d = np.array(face3d, dtype=np.float64)

        camera = Camera()
        success, rot_vec, trans_vec = cv2.solvePnP(face3d,
                                                    face2d,
                                                    camera.internal_matrix,
                                                    camera.distortion_matrix,
                                                    flags=cv2.SOLVEPNP_ITERATIVE)
        
        rmat = cv2.Rodrigues(rot_vec)[0]

        P = np.hstack((rmat, np.zeros((3, 1), dtype=np.float64)))
        eulerAngles =  cv2.decomposeProjectionMatrix(P)[6]
        yaw = eulerAngles[1, 0]
        pitch = eulerAngles[0, 0]
        roll = eulerAngles[2,0]
        
        if pitch < 0:
            pitch = - 180 - pitch
        elif pitch >= 0: 
            pitch = 180 - pitch
        
        yaw *= -1
        pitch *= -1
        
        # if nose2d:
        #     nose2d = nose2d
        #     p1 = (int(nose2d[0]), int(nose2d[1]))
        #     p2 = (int(nose2d[0] - yaw * 2), int(nose2d[1] - pitch * 2))
        
        return yaw, pitch, roll 

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

def get_head_pose(image_path):
    """
    Given an image, estimate the head pose (roll, pitch, yaw angles).

    Args:
        image: Image to estimate head pose.

    Returns:
        tuple: Roll, Pitch, Yaw angles if face landmarks are detected, otherwise None.
    """

    image = cv2.imread(image_path)
    # Convert the image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect face landmarks.
    results = face_mesh.process(image_rgb)

    img_h, img_w, _ = image.shape
    face_3d = []
    face_2d = []


    if results.multi_face_landmarks:       
        for face_landmarks in results.multi_face_landmarks:
            key_landmark_positions=[]
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in HEAD_POSE_LANDMARKS:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

                    landmark_position = [x,y]
                    key_landmark_positions.append(landmark_position)
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
            yaw, pitch, roll = calculate_pose(key_landmark_positions)
            print(f'Roll: {roll:.4f}, Pitch: {pitch:.4f}, Yaw: {yaw:.4f}')
            draw_axis(image, yaw, pitch, roll)
            debug_image_path = image_path.replace('.jpg', '_debug.jpg')  # Modify as needed
            cv2.imwrite(debug_image_path, image)
            print(f'Debug image saved to {debug_image_path}')
            
            return roll, pitch, yaw 

    return None




def get_head_pose_velocities_at_frame(video_reader:VideoReader, frame_index, n_previous_frames=2):

    # Adjust frame_index if it's larger than the total number of frames
    total_frames = len(video_reader)
    frame_index = min(frame_index, total_frames - 1)

    # Calculate starting index for previous frames
    start_index = max(0, frame_index - n_previous_frames)

    head_poses = []
    for idx in range(start_index, frame_index + 1):
        # idx is the frame index you want to access
        frame_tensor = video_reader[idx]

        #  check emodataset decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        # Assert that frame_tensor is a PyTorch tensor
        assert isinstance(frame_tensor, torch.Tensor), "Expected a PyTorch tensor"

        image = video_reader[idx].numpy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:       
            for face_landmarks in results.multi_face_landmarks:
                key_landmark_positions=[]
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in HEAD_POSE_LANDMARKS:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                        landmark_position = [x,y]
                        key_landmark_positions.append(landmark_position)
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
                yaw, pitch, roll = calculate_pose(key_landmark_positions)
                head_poses.append((roll, pitch, yaw))

    # Calculate velocities
    head_velocities = []
    for i in range(len(head_poses) - 1):
        roll_diff = head_poses[i + 1][0] - head_poses[i][0]
        pitch_diff = head_poses[i + 1][1] - head_poses[i][1]
        yaw_diff = head_poses[i + 1][2] - head_poses[i][2]
        head_velocities.append((roll_diff, pitch_diff, yaw_diff))

    return head_velocities
# # Usage example
# image_path = 'frame_0003.jpg'  # Replace with your image path
# head_pose = get_head_pose(image_path)
# if head_pose:
#     roll, pitch, yaw = head_pose
#     print(f'Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}')
# else:
#     print('No face detected or the face landmarks are not sufficient for pose estimation.')
