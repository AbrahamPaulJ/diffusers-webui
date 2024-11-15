from PIL import Image, ImageOps
import numpy as np
import cv2
import torch


def retrieve_mask(mode = "Inpaint", outpaint_img_pos= "Center", image= None):
    if image is None:
        # Return a blank image if there's no input image
        return Image.new("RGBA", (520, 520), (0, 0, 0, 0))

    if mode == "Inpaint":
        # For Inpaint mode, check if height exceeds 450px
        width, height = image.size
        if height > 450:
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = 450 / height
            new_width = int(width * scale_factor)
            new_height = 450
            # Resize the image with the new dimensions
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image  # Return resized image for Inpaint mode

    elif mode == "Outpaint":
        # For Outpaint mode, resize the image so that the larger side is 200
        width, height = image.size
        if width > height:
            scale_factor = 200 / width
            new_width = 200
            new_height = int(height * scale_factor)
        else:
            scale_factor = 200 / height
            new_height = 200
            new_width = int(width * scale_factor)

        # Resize the image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a black image of 500x500
        final_image = Image.new("RGBA", (500, 500), (0, 0, 0, 255))

        # Calculate offsets based on `outpaint_img_pos`
        if outpaint_img_pos == "Center":
            x_offset = (500 - new_width) // 2
            y_offset = (500 - new_height) // 2
        elif outpaint_img_pos == "Top":
            x_offset = (500 - new_width) // 2
            y_offset = 0  # Align to top center
        elif outpaint_img_pos == "Bottom":
            x_offset = (500 - new_width) // 2
            y_offset = 500 - new_height  # Align to bottom center

        # Paste the resized image onto the black background at the calculated position
        final_image.paste(image, (x_offset, y_offset))

        return final_image  # Return the combined image for Outpaint mode


def use_brush(image):
    # print("use_brush function was accessed.")
    layers = image["layers"]
    brush_mask = layers[0]

    # Convert brush_mask to a numpy array
    brush_mask = np.array(brush_mask)

    # # Check if brush_mask is empty or has no drawing
    # if np.all(brush_mask == 0):  # Assuming 0 means no drawing (black/transparent)
    #     # Create a black RGBA background with the same size
    #     final_mask = Image.new("RGB", (brush_mask.shape[1], brush_mask.shape[0]), "BLACK")
    #     return final_mask

    # Create a white RGBA background
    white_background = Image.new("RGBA", brush_mask.shape[1::-1], "WHITE")
    # Convert brush_mask to PIL Image
    brush_mask_image = Image.fromarray(brush_mask)
    # Paste the brush mask onto the white background
    white_background.paste(brush_mask_image, (0, 0), brush_mask_image)
    # Convert the result to RGB and save as final_mask
    final_mask = white_background.convert('RGB')
    return final_mask

def use_crop(image):
    return image["background"]

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

def add_padding(image, target_width, target_height, pad_color=None):
    """Resize and pad image to target dimensions while maintaining aspect ratio."""
    
    # Check if image is empty
    if image is None or image.size == (0, 0):
        raise ValueError("Input image is empty or not valid.")

    img_width, img_height = image.size
    # print(f"Original image size: {img_width}x{img_height}")  # Debugging line
    target_aspect_ratio = target_width / target_height
    current_aspect_ratio = img_width / img_height

    # Calculate new dimensions and padding
    if current_aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / current_aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * current_aspect_ratio)

    # Resize the image to fit within the target dimensions
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Check the new image size after resizing
    # print(f"Resized image size (before padding): {image.size}")  # Debugging line
    new_img_width, new_img_height = image.size

    # Calculate padding to reach target dimensions
    padding = (
        (target_width - new_img_width) // 2,  # Left padding
        (target_height - new_img_height) // 2,  # Top padding
        (target_width - new_img_width + 1) // 2,  # Right padding
        (target_height - new_img_height + 1) // 2   # Bottom padding
    )

    # If no padding color is specified, calculate from edges of the image
    if pad_color is None:
        # Ensure the image has dimensions for edge color calculations
        if new_img_width > 0 and new_img_height > 0:
            # Get edge colors if the image has valid dimensions
            left_color = image.getpixel((0, new_img_height // 2))  # Left edge
            right_color = image.getpixel((new_img_width - 1, new_img_height // 2))  # Right edge
            top_color = image.getpixel((new_img_width // 2, 0))  # Top edge
            bottom_color = image.getpixel((new_img_width // 2, new_img_height - 1))  # Bottom edge
            
            edge_colors = [left_color, right_color, top_color, bottom_color]
            pad_color = tuple(sum(c) // len(c) for c in zip(*edge_colors))
        else:
            pad_color = (128, 128, 128)  # Default color if image is too small

    # Ensure the pad_color is appropriate for the image mode
    return ImageOps.expand(image, padding, pad_color)

def auto_select_dimensions(image, target_sizes=[512, 768, 1024]):
    # Get the original dimensions of the image
    original_width, original_height = image.size
    
    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Determine the closest target size for width and height
    if aspect_ratio > 1:  # Landscape
        target_width = min(target_sizes, key=lambda x: abs(x - original_width))
        target_height = int(target_width / aspect_ratio)
    else:  # Portrait or Square
        target_height = min(target_sizes, key=lambda x: abs(x - original_height))
        target_width = int(target_height * aspect_ratio)

    # Round the width and height to the nearest target size
    target_width = min(target_sizes, key=lambda x: abs(x - target_width))
    target_height = min(target_sizes, key=lambda x: abs(x - target_height))

    return target_width, target_height

def resize_to_max_side(image, target_size=1024):
    """
    Resizes the image so that the larger side becomes `target_size` (e.g., 1024),
    maintaining the aspect ratio.

    Args:
        image (PIL.Image): The input image.
        target_size (int): The target size for the larger dimension.

    Returns:
        PIL.Image: The resized image with the larger side set to `target_size`.
    """
    width, height = image.size
    
    # Determine scale factor to make the larger side equal to target_size
    if width > height:
        scale_factor = target_size / width
    else:
        scale_factor = target_size / height

    # Calculate new dimensions and resize the image
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

def get_inner_dimensions(image):
    # Convert the image to grayscale and then to binary (black/white)
    gray_image = image.convert("L")  # Convert to grayscale
    binary_mask = gray_image.point(lambda p: p > 0 and 255)  # Create a binary mask

    # Find the bounding box of the non-black areas
    bbox = binary_mask.getbbox()  # Returns (left, upper, right, lower)

    if bbox is None:
        return None  # No non-black pixels found

    # Calculate the width and height based on the bounding box
    inner_width = bbox[2] - bbox[0]  # right - left
    inner_height = bbox[3] - bbox[1]  # lower - upper
    print(f"Inner dimensions are: {inner_width} x {inner_height}.")
    return inner_width, inner_height

def get_bounding_box_sides(outpaint_mask):
    # Convert the image to grayscale and create a binary mask to isolate non-black areas
    gray_image = outpaint_mask.convert("L")  # Convert to grayscale
    binary_mask = gray_image.point(lambda p: p > 0 and 255)  # Convert to binary (black/white)

    # Get the bounding box of the non-black region
    bbox = binary_mask.getbbox()  # Returns (left, upper, right, lower)
    
    if bbox is not None:
        # Calculate the lengths of each side of the bounding box
        left_side = bbox[0]
        right_side = outpaint_mask.width - bbox[2]
        top_side = bbox[1]
        bottom_side = outpaint_mask.height - bbox[3]

        # Print the results
        # print(f"Left side: {left_side} pixels")
        # print(f"Right side: {right_side} pixels")
        # print(f"Top side: {top_side} pixels")
        # print(f"Bottom side: {bottom_side} pixels")
        
        # Return the side lengths as a tuple
        return (left_side, right_side, top_side, bottom_side)
    else:
        print("No non-black region found.")
        return (0, 0, 0, 0)  # Return zeros if no non-black region is found

def resize_to_nearest_multiple_of_8(image):
    # Get the current dimensions of the image
    width, height = image.size
    
    # Calculate the nearest dimensions that are multiples of 8
    new_width = (width + 7) // 8 * 8
    new_height = (height + 7) // 8 * 8

    # Resize the image to these dimensions
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

import cv2
import torch
from torchvision import transforms
from PIL import Image

def remove_background(image: Image.Image, threshold: int = 200) -> Image.Image:
    image = image.convert("RGBA")
    data = image.getdata()
    new_data = []
    for item in data:
        avg = sum(item[:3]) / 3
        if avg > threshold:
            new_data.append((0, 0, 0, 0))  # Make white pixels transparent
        else:
            new_data.append(item)  # Keep other colors

    image.putdata(new_data)
    return image

def create_composite(background: Image.Image, layer: Image.Image) -> Image.Image:
    # Ensure both images are in RGBA mode to handle transparency
    background = background.convert("RGBA")
    layer = layer.convert("RGBA")
    
    # Create a copy of the background to avoid modifying the original
    composite = background.copy()
    
    # Paste the layer onto the background
    composite.paste(layer, (0, 0), layer)  # Use layer as its own mask to handle transparency
    
    return composite    
