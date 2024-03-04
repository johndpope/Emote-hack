import cv2
import mediapipe as mp
import numpy as np
from face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
import argparse

def direction_based_on_angle(y):
    if y > 20:
        if y < 45:
            text = "Looking Left"
        else:
            text = "Looking Back Left"
    elif y < -20:
        if y > -45:
            text = "Looking Right"
        else:
            text = "Looking Back Left"
    else:
        text = "Forward"
    return text


# if video == True and video_path is provided it will analyze the video. Otherwise it will capture the webcam.
def webcam(video=False, video_path="", out_path="mediapipe.avi"):
    # For webcam input:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (1080, 1080))

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    if not video:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                if video:
                    cap.release()
                    exit()
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape

            # Point to use for estimation
            points_idx = [33, 263, 61, 291, 199]
            points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
            points_idx = list(set(points_idx))
            points_idx.sort()

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            pcf = PCF(
                near=1,
                far=10000,
                frame_height=img_h,
                frame_width=img_w,
                fy=cam_matrix[1, 1],
            )
            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            multi_face_landmarks = results.multi_face_landmarks
            if results.multi_face_landmarks:
                face_landmarks = multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                landmarks = landmarks.T
                metric_landmarks, pose_transform_mat = get_metric_landmarks(
                    landmarks.copy(), pcf
                )
                model_points = metric_landmarks[0:3, points_idx].T
                image_points = (
                        landmarks[0:2, points_idx].T
                        * np.array([img_w, img_h])[None, :]
                )
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points,
                    image_points,
                    cam_matrix,
                    dist_matrix,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

                (nose_end_point2D, jacobian) = cv2.projectPoints(
                    np.array([(0.0, 0.0, 25.0)]),
                    rotation_vector,
                    translation_vector,
                    cam_matrix,
                    dist_matrix,
                )
                # mp_drawing = mp.solutions.drawing_utils
                # mp_face_mesh = mp.solutions.face_mesh
                for face_landmarks in multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                image = cv2.line(image, p1, p2, (255, 0, 0), 2)
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rotation_vector)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0]
                y = angles[1]
                text = direction_based_on_angle(y)
                cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                cv2.putText(image, "Yaw Angle: " + str(round(angles[1])), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                if not video:
                    cv2.imshow('MediaPipe Face Mesh', image)

                if not video:
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                else:
                    out.write(image)
    if video:
        out.release()
    cap.release()
    print("DONE")

# For image_evaluation images:
def image_evaluation(path, debug=True, image=None):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5) as face_mesh:
        # Read image
        if image is None:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if debug:
            image.flags.writeable = True
        else:
            image.flags.writeable = False
        # Convert BGR to RGB
        results = face_mesh.process(image)


        img_h, img_w, img_c = image.shape

        # Point to use for estimation

        points_idx = [33, 263, 61, 291, 199]
        # To use all the available points
        points_idx = [x for x in range(0, 468)]
        points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
        points_idx = list(set(points_idx))
        points_idx.sort()
        # The camera matrix
        focal_length = 1 * img_w

        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])

        pcf = PCF(
            near=1,
            far=10000,
            frame_height=img_h,
            frame_width=img_w,
            fy=cam_matrix[1, 1],
        )
        # The Distance Matrix
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        multi_face_landmarks = results.multi_face_landmarks
        if results.multi_face_landmarks:
            face_landmarks = multi_face_landmarks[0]
            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            )
            landmarks = landmarks.T
            metric_landmarks, pose_transform_mat = get_metric_landmarks(
                landmarks.copy(), pcf
            )
            model_points = metric_landmarks[0:3, points_idx].T
            image_points = (
                    landmarks[0:2, points_idx].T
                    * np.array([img_w, img_h])[None, :]
            )
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                cam_matrix,
                dist_matrix,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if debug:
                (nose_end_point2D, jacobian) = cv2.projectPoints(
                    np.array([(0.0, 0.0, 25.0)]),
                    rotation_vector,
                    translation_vector,
                    cam_matrix,
                    dist_matrix,
                )
                # mp_drawing = mp.solutions.drawing_utils
                # mp_face_mesh = mp.solutions.face_mesh
                for face_landmarks in multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                image = cv2.line(image, p1, p2, (255, 0, 0), 2)
            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rmat, translation_vector))
            # Get angles ([0] -> pitch, [1] -> yaw, [2] -> roll)
            angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        else:
            return [-1, -1, -1]
    # Return pitch, yaw, roll
    print(f'Pitch: {angles[0][0]}, Yaw: {angles[1][0]}, Roll: {angles[2][0]}')
    return [angles[0][0], angles[1][0], angles[2][0]]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Head Pose Estimator")
    parser.add_argument("--image", type=bool, default=False, help="Set to true to process an image")
    parser.add_argument("--image_path", type=str, default=False, help="Path to image")
    parser.add_argument("--video", type=bool, default=False, help="Set to true to process a video")
    parser.add_argument("--video_path", type=str, required=False, help="Path to video")
    parser.add_argument("--out_path", type=str, required=False, help="Output video path")

    args = parser.parse_args()

    if args.video:
        webcam(video=args.video, video_path=args.video_path, out_path=args.out_path)
    elif args.image:
        print(image_evaluation(args.image_path))
    else:
        webcam()

