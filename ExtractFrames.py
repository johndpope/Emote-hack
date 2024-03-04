import cv2
from PIL import Image
import json
import os

# Your JSON data for the video clip
clip_data = {
    "M2Ohb0FAaJU_1": {
        "ytb_id": "M2Ohb0FAaJU",
        "duration": {"start_sec": 81.62, "end_sec": 86.17},
        "bbox": {"top": 0.0, "bottom": 0.8815, "left": 0.1964, "right": 0.6922},
        "attributes": {
            "appearance": [0, 0, 1],  # Truncated for example purposes
            "action": [0, 0, 0],      # Truncated for example purposes
            "emotion": {"sep_flag": False, "labels": "neutral"}
        },
        "version": "v0.1"
    }
}

# Define the function to extract frames
def extract_frames(video_path, clip_info):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    start_frame = int(clip_info['duration']['start_sec'] * fps)
    end_frame = int(clip_info['duration']['end_sec'] * fps)
    bbox = clip_info['bbox']
    
    # Set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if frames cannot be read
        
        # Convert to PIL image for easier cropping
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Calculate bounding box coordinates
        width, height = frame.size
        left = bbox['left'] * width
        top = bbox['top'] * height
        right = bbox['right'] * width
        bottom = bbox['bottom'] * height
        frame = frame.crop((left, top, right, bottom))
        
        # Save the frame with bounding box applied
        frame.save(f"frame_{frame_num}.png")

    cap.release()


def extract_and_save_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Total frames extracted: {frame_count}")

# Assuming your video is named 'M2Ohb0FAaJU_1.mp4' and located in the current directory
video_path = 'M2Ohb0FAaJU_1.mp4'
extract_and_save_frames(video_path,'.')