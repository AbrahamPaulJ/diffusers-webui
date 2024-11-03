import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def upscale(image, scale_factor=4):
    """Upscales the input image using ESRGAN with a given scale factor."""
    # Ensure scale_factor is of correct type
    scale_factor = int(scale_factor)

    # Load the appropriate model based on the scale factor
    model = RealESRGAN(device, scale=scale_factor)
    model.load_weights(r'models\RealESRGAN_x4.pth', download=True)  # Adjust path if needed
    
        # Check if `image` is a list
    if isinstance(image, list):
        image = image[0]  # Take the first image if it's a list
        if isinstance(image, tuple):
            image = image[0]  # Take the first item if it's a tuple

    # Convert the input image to RGB
    image = image.convert("RGB")

    # Print the input image size for debugging
    print("Input image size:", image.size)

    # Upscale the image using the model
    output_image = model.predict(image)

    # Convert output image to a NumPy array
    output_image_np = np.array(output_image)

    # Clip values to be in the valid range [0, 255] and convert to uint8
    output_image_np = np.clip(output_image_np, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    output_image = Image.fromarray(output_image_np)
    output_image.save("upscaled_image.jpg", "JPEG")

    return output_image
