import torch
from torchvision import transforms
from PIL import Image as PILImage
import cv2
import numpy as np

from model import RepNet6D
import utils

# Assume 'utils.py' contains necessary transformation functions and drawing functions.
# Replace 'backbone_file.pth' and 'model_weights.pth' with your actual file paths.
backbone_file = 'backbone_file.pth'
model_weights = 'model_weights.pth'

# Initialize the transformation for input images
transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize the model
model = RepNet6D(backbone_name='RepVGG-B1g4',
                 backbone_file=backbone_file,
                 deploy=True,
                 pretrained=False)

# Load the saved weights into the model
model.load_state_dict(torch.load(model_weights))
model.eval()  # Set the model to evaluation mode

# Load an image
image_path = 'path_to_your_image.jpg'  # Replace with your image path
image = PILImage.open(image_path).convert('RGB')

# Transform the image
image_tensor = transformations(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    rotation_matrix = model(image_tensor)
    euler_angles = utils.compute_euler_angles_from_rotation_matrices(rotation_matrix) * 180 / np.pi

# Convert the Euler angles to pitch, yaw, roll
pitch, yaw, roll = euler_angles.squeeze().tolist()  # Assuming euler_angles is a 1D tensor

# Draw the axis on the image using the utility function
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
utils.draw_axis(image_cv, yaw, pitch, roll, tdx=image_cv.shape[1] // 2, tdy=image_cv.shape[0] // 2, size=100)  # tdx and tdy are the center of the image

# Display the image
cv2.imshow('Head Pose Estimation', image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
