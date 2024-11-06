from PIL import Image
import numpy as np
import cv2
import torch

def create_control_image(input_image, controlnet_type):
    """Generate control images for ControlNet (canny, depth, etc.) based on controlnet_type."""
    
    # Convert input_image to grayscale if needed for certain operations
    if isinstance(input_image, Image.Image):
        input_image_np = np.array(input_image.convert("RGB"))
    else:
        print("Error: input_image is not a PIL Image.")
        return None

    control_image = None
    controlnet_type = controlnet_type.lower()  # Normalize for case-insensitive matching
    print(f"ControlNet type: {controlnet_type}")  # Debugging

    # Process based on the controlnet type
    if "canny" in controlnet_type:
        # Convert the image to grayscale for edge detection
        gray_image = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)
        print("Applying Canny edge detection...")  # Debugging
        # Apply canny edge detection
        control_image = cv2.Canny(gray_image, 100, 200)
        control_image = Image.fromarray(control_image)
        return control_image  # Return after canny to prevent overwrites

    if "depth" in controlnet_type:
        print("Applying Depth estimation...")  # Debugging
        from transformers import pipeline
        depth_estimator = pipeline('depth-estimation', device=0 if torch.cuda.is_available() else "cpu")
        image = depth_estimator(input_image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)
        return control_image

    if "openpose" in controlnet_type:
        print("Applying OpenPose detection...")  # Debugging
        from controlnet_aux import OpenposeDetector
        processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        control_image = processor(input_image, hand_and_face=True)
        return control_image

    # Resize control_image to match input dimensions
    if control_image and (control_image.size != input_image.size):
        control_image = control_image.resize(input_image.size, Image.BILINEAR)

    return control_image