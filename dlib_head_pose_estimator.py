import math
import time

import numpy
import cv2
import sys
import os
import numpy as np
import dlib
import argparse
from mediapipe_facemesh_head_pose_estimator import direction_based_on_angle
class faceLandmarkDetection:
    def __init__(self, landmarkPath):
        # Check if the file provided exist
        if (os.path.isfile(landmarkPath) == False):
            raise ValueError('haarCascade: the files specified do not exist.')

        self._predictor = dlib.shape_predictor(landmarkPath)

    ##
    # Find landmarks in the image provided
    # @param inputImg the image where the algorithm will be called
    #
    def returnLandmarks(self, inputImg, roiX, roiY, roiW, roiH, points_to_return=range(0, 68)):
        # Creating a dlib rectangle and finding the landmarks
        dlib_rectangle = dlib.rectangle(left=int(roiX), top=int(roiY), right=int(roiW), bottom=int(roiH))
        dlib_landmarks = self._predictor(inputImg, dlib_rectangle)
        # It selects only the landmarks that
        # have been indicated in the input parameter "points_to_return".
        # It can be used in solvePnP() to estimate the 3D pose.
        self._landmarks = numpy.zeros((len(points_to_return), 2), dtype=numpy.float32)
        counter = 0
        for point in points_to_return:
            self._landmarks[counter] = [dlib_landmarks.parts()[point].x, dlib_landmarks.parts()[point].y]
            counter += 1
        return self._landmarks


# Antropometric constant values of the human head.
# Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
# X-Y-Z with X pointing forward and Y on the left.
# The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = numpy.float32([-100.0, -77.5, -5.0])  # 0
P3D_GONION_RIGHT = numpy.float32([-110.0, -77.5, -85.0])  # 4
P3D_MENTON = numpy.float32([0.0, 0.0, -122.7])  # 8
P3D_GONION_LEFT = numpy.float32([-110.0, 77.5, -85.0])  # 12
P3D_LEFT_SIDE = numpy.float32([-100.0, 77.5, -5.0])  # 16
P3D_FRONTAL_BREADTH_RIGHT = numpy.float32([-20.0, -56.1, 10.0])  # 17
P3D_FRONTAL_BREADTH_LEFT = numpy.float32([-20.0, 56.1, 10.0])  # 26
P3D_SELLION = numpy.float32([0.0, 0.0, 0.0])  # 27
P3D_NOSE = numpy.float32([21.1, 0.0, -48.0])  # 30
P3D_SUB_NOSE = numpy.float32([5.0, 0.0, -52.0])  # 33
P3D_RIGHT_EYE = numpy.float32([-20.0, -65.5, -5.0])  # 36
P3D_RIGHT_TEAR = numpy.float32([-10.0, -40.5, -5.0])  # 39
P3D_LEFT_TEAR = numpy.float32([-10.0, 40.5, -5.0])  # 42
P3D_LEFT_EYE = numpy.float32([-20.0, 65.5, -5.0])  # 45
# P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
# P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0])  # 62

# The points to track
# These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0, 68))  # Used for debug only
# reduce the video resolution (100 is full resolution)



def webcam(video=False, video_path="./test_video.mp4", out_path="./dlib.avi", scale_percent = 80):
    # Check if some argumentshave been passed
    # pass the path of a video
    if video:
        if (os.path.isfile(video_path) == False):
            print("File does not exist")
            return
        else:
            video_capture = cv2.VideoCapture(video_path)
            # define video writer codec
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (
                int(video_capture.get(3)), int(video_capture.get(4))))
    else:
        video_capture = cv2.VideoCapture(-1)
    # Camera dimensions
    cam_w = int(video_capture.get(3) * scale_percent / 100)
    cam_h = int(video_capture.get(4) * scale_percent / 100)

    # c_x and c_y are the optical centers
    c_x = cam_w / 2
    c_y = cam_h / 2
    # f_x in f_y are the focal lengths
    f_x = c_x / numpy.tan(60 / 2 * numpy.pi / 180)
    f_y = f_x

    # Estimated camera matrix values
    camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y],
                                   [0.0, 0.0, 1.0]])

    # Distortion coefficients
    dist_coeffs = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # This matrix contains the 3D points of the
    # 11 landmarks we want to find. It has been
    # obtained from antrophometric measurement
    # on the human head.
    landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])

    # Declaring the two classifiers
    # my_cascade = haarCascade("../etc/haarcascade_frontalface_alt.xml", "../etc/haarcascade_profileface.xml")
    dlib_landmarks_file = "./shape_predictor_68_face_landmarks.dat"
    if (os.path.isfile(dlib_landmarks_file) == False):
        print("The dlib landmarks file is missing! Use the following commands to download and unzip: ")
        print(">> wget dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(">> bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
        return

    face_landmark_detector = faceLandmarkDetection(dlib_landmarks_file)
    # frontal_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat-1")
    frontal_face_detector = dlib.get_frontal_face_detector()

    while (True):
        ret, frame = video_capture.read()
        # Capture frame-by-frame
        if frame is None:
            break
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        try:
            pos = frontal_face_detector(frame, 1)[0]
        except IndexError:
            print("no face")
            continue
        face_x1 = pos.left()
        face_y1 = pos.top()
        face_x2 = pos.right()
        face_y2 = pos.bottom()
        text_x1 = face_x1
        text_y1 = face_y1 - 3

        cv2.putText(frame, "FACE", (text_x1, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame,
                      (face_x1, face_y1),
                      (face_x2, face_y2),
                      (0, 255, 0),
                      2)

        landmarks_2D = face_landmark_detector.returnLandmarks(frame, face_x1, face_y1, face_x2, face_y2,
                                                              points_to_return=TRACKED_POINTS)

        for point in landmarks_2D:
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

        # Applying the PnP solver to find the 3D pose
        # of the head from the 2D position of the
        # landmarks.
        # retval - bool
        # rotation_vector - Output rotation vector that, together with translation_vector, brings
        # points from the model coordinate system to the camera coordinate system.
        # translation_vector - Output translation vector.
        retval, rotation_vector, translation_vector = cv2.solvePnP(landmarks_3D,
                                                                   landmarks_2D,
                                                                   camera_matrix, dist_coeffs,
                                                                   flags=cv2.SOLVEPNP_ITERATIVE)

        # Project 3D points onto the image plane
        axis = numpy.float32([[50, 0, 0],
                              [0, 50, 0],
                              [0, 0, 50]])
        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        rvec_matrix, _ = cv2.Rodrigues(rotation_vector)

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rvec_matrix)
        # Draw axis on image
        pt2 = [imgpts[0].ravel(), imgpts[1].ravel(), imgpts[2].ravel()]
        sellion_xy = (int(landmarks_2D[7][0]), int(landmarks_2D[7][1]))
        cv2.line(frame, sellion_xy, (int(pt2[1][0]), int(pt2[1][1])), (0, 255, 0), 3)  # GREEN
        cv2.line(frame, sellion_xy, (int(pt2[2][0]), int(pt2[2][1])), (255, 0, 0), 3)  # BLUE
        cv2.line(frame, sellion_xy, (int(pt2[0][0]), int(pt2[0][1])), (0, 0, 255), 3)  # RED

        # Writing in the output file
        y = angles[1] - 80
        print(angles)
        if y > 10:
            if y < 20:
                text = "Looking Left"
            else:
                text = "Looking Back Left"
        elif y < -10:
            if y > -20:
                text = "Looking Right"
            else:
                text = "Looking Back Left"
        else:
            text = "Forward"
        cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Yaw Angle: " + str(round(angles[1]-80)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    2)
        frame = cv2.resize(frame, (int(cam_w * 100 / scale_percent), int(cam_h * 100 / scale_percent)), interpolation=cv2.INTER_AREA)
        if video:
            out.write(frame)

        # Showing the frame and waiting
        # for the exit command
        if not video:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera
    video_capture.release()
    if video:
        out.release()
    print("DONE")


def image_evaluation(path, debug=True, image=None):
    if image is None:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Camera dimensions
    cam_h, cam_w, channels = image.shape
    # c_x and c_y are the optical centers
    c_x = cam_w / 2
    c_y = cam_h / 2
    # f_x in f_y are the focal lengths
    f_x = c_x / numpy.tan(60 / 2 * numpy.pi / 180)
    f_y = f_x

    # Estimated camera matrix values
    camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y],
                                   [0.0, 0.0, 1.0]])

    # Distortion coefficients
    dist_coeffs = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # This matrix contains the 3D points of the
    # 11 landmarks we want to find. It has been
    # obtained from antrophometric measurement
    # on the human head.
    landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])
    # Declaring the two classifiers

    dlib_landmarks_file = "./shape_predictor_68_face_landmarks.dat"
    if (os.path.isfile(dlib_landmarks_file) == False):
        print("The dlib landmarks file is missing! Use the following commands to download and unzip: ")
        print(">> wget dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(">> bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
        return

    face_landmark_detector = faceLandmarkDetection(dlib_landmarks_file)
    # frontal_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat-1")
    frontal_face_detector = dlib.get_frontal_face_detector()
    end = time.time()
    try:
        pos = frontal_face_detector(image, 1)[0]
    except IndexError:
        return -1, -1, -1

    face_x1 = pos.left()
    face_y1 = pos.top()
    face_x2 = pos.right()
    face_y2 = pos.bottom()
    text_x1 = face_x1
    text_y1 = face_y1 - 3

    landmarks_2D = face_landmark_detector.returnLandmarks(image, face_x1, face_y1, face_x2, face_y2,
                                                          points_to_return=TRACKED_POINTS)

    retval, rotation_vector, translation_vector = cv2.solvePnP(landmarks_3D,
                                                               landmarks_2D,
                                                               camera_matrix, dist_coeffs)

    # Project 3D points onto the image plane
    axis = numpy.float32([[50, 0, 0],
                          [0, 50, 0],
                          [0, 0, 50]])
    # imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    # modelpts, jac2 = cv2.projectPoints(landmarks_3D, rotation_vector, translation_vector, camera_matrix,
    #                                    dist_coeffs)
    rvec_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = numpy.hstack((rvec_matrix, translation_vector))

    angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    roll, pitch, yaw = angles  # Angles are in radians
    if debug:
        cv2.putText(image, "FACE ", (text_x1, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(image,
                      (face_x1, face_y1),
                      (face_x2, face_y2),
                      (0, 255, 0),
                      2)
        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        pt2 = [imgpts[0].ravel(), imgpts[1].ravel(), imgpts[2].ravel()]
        sellion_xy = (int(landmarks_2D[7][0]), int(landmarks_2D[7][1]))
        cv2.line(image, sellion_xy, (int(pt2[1][0]), int(pt2[1][1])), (0, 255, 0), 3)  # pitch GREEN
        cv2.line(image, sellion_xy, (int(pt2[2][0]), int(pt2[2][1])), (255, 0, 0), 3)  # yaw BLUE
        cv2.line(image, sellion_xy, (int(pt2[0][0]), int(pt2[0][1])), (0, 0, 255), 3)  # roll RED
        debug_image_path = 'debug_image.jpg'  # Specify your path and filename
        cv2.imwrite(debug_image_path, image)
        print(f"Debug image saved at: {debug_image_path}")
    return [angles[0][0], angles[1][0], angles[2][0]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head Pose Estimator")
    parser.add_argument("--image", type=bool, default=False, help="Set to true to process an image")
    parser.add_argument("--image_path", type=str, default=False, help="Path to image")
    parser.add_argument("--video", type=bool, default=False, help="Set to true to process a video")
    parser.add_argument("--video_path", type=str, required=False, help="Path to video")
    parser.add_argument("--out_path", type=str, required=False, help="Output video path")
    parser.add_argument("--scale_percent", type=int, required=False, help="Process in x% of original video size")
    args = parser.parse_args()

    if args.video:
        webcam(video=args.video, video_path=args.video_path, out_path=args.out_path, scale_percent=args.scale_percent)
    elif args.image:
        print(image_evaluation(args.image_path))
    else:
        webcam(scale_percent=args.scale_percent)