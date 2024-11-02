# modules/inpainting.py

import torch
from pipelines import PipelineManager
from PIL import Image, ImageOps, PngImagePlugin, ImageFilter
import numpy as np
from datetime import datetime
import os
from modules.manage_models import model_dir

def reset_brush(image):
    if image is None:
        return Image.new("RGBA", (520, 520), (0, 0, 0, 0))  # Return a blank image if no input
    width, height = image.size
    # Check if height exceeds 550px
    if height > 450:
        # Calculate the scaling factor to maintain aspect ratio
        scale_factor = 450 / height
        new_width = int(width * scale_factor)
        new_height = 450
        # Resize the image with the new dimensions
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image

def use_brush(image):
    if image is None:
        return Image.new("RGBA", (520, 520), (255, 255, 255, 255))  # Return a white image if no input
    print("use_brush function was accessed.")
    # Extract layers from the image editor value
    layers = image["layers"]
    # Create the brush mask from the first layer (assuming brush strokes are stored here)
    brush_mask = layers[0] if layers and layers[0] is not None else None
    # If there's no brush mask, return a blank white image
    if brush_mask is None:
        return Image.new("RGBA", (520, 520), (255, 255, 255, 255))  # White image as fallback
    # Convert brush_mask to a numpy array
    brush_mask = np.array(brush_mask)
    # Create a white RGBA background
    white_background = Image.new("RGBA", brush_mask.shape[1::-1], "WHITE")
    # Convert brush_mask to PIL Image
    brush_mask_image = Image.fromarray(brush_mask)
    # Paste the brush mask onto the white background
    white_background.paste(brush_mask_image, (0, 0), brush_mask_image)
    # Convert the result to RGB and save as final_mask
    final_mask = white_background.convert('RGB')
    return final_mask

def generate_inpaint_image(pipeline_manager: PipelineManager,checkpoint, scheduler, use_controlnet, seed, generator, prompt, negative_prompt, width, height, steps, cfg_scale, clip_skip, mask_output, fill_setting, input_image, segment_type, maintain_aspect_ratio, post_process, custom_dimensions, denoise_strength, batch_size, mask_blur):
    """Generate an inpainting image using the loaded pipeline."""
    pipe = pipeline_manager.active_pipe  # Use the active pipeline

    if pipe is None:
        raise ValueError("Inpainting pipeline not initialized.")
  
    if not custom_dimensions:
        width, height = auto_select_dimensions(input_image)
        print(f"Target dimensions are {height}x{width}")

    if segment_type == "Use Brush":
        mask_output = use_brush(mask_output)
        mask_output = mask_output.resize((input_image.width, input_image.height), Image.Resampling.LANCZOS) 
    else:
        mask_output = mask_output["background"]

    # Convert the mask image to grayscale
    mask_output = mask_output.convert("L")

    if maintain_aspect_ratio:
        mask_output = add_padding(mask_output, width, height, pad_color=0)  # Black padding for the mask
        input_image = add_padding(input_image, width, height)  # Use average color for input image
    
    if mask_blur > 0:    
        mask_output = mask_output.filter(ImageFilter.GaussianBlur(mask_blur))
        
    mask_array = np.array(mask_output) / 255.0

    mask_image = mask_array if fill_setting == "Generate Inside Mask" else 1.0 - mask_array
    mask_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
    mask_pil.save("maskimg.jpg")

    eta = 1.0

    # Prepare the keyword arguments for the pipeline
    pipe_kwargs = {
        "num_images_per_prompt": batch_size,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "generator": generator,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "strength": denoise_strength,
        "guidance_scale": cfg_scale,
        "clip_skip": clip_skip,
        "mask_image": 1.0 - mask_image,  # Use the determined mask
        "image": input_image,
    }

    # Add control_image to the keyword arguments if ControlNet is used
    if use_controlnet:
            control_image = make_inpaint_condition(input_image, mask_pil) if use_controlnet else None
            control_img_view = Image.fromarray((control_image.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            control_img_view.save("controlimg.jpg")
            
            pipe_kwargs["control_image"] = control_image
            if scheduler=="DDIMScheduler":
                pipe_kwargs["eta"] = eta
            

    # Generate the image with the prepared arguments
    generated_images = pipe(**pipe_kwargs).images

    processed_images = []

    for generated_image in generated_images:
        if post_process:
            # Convert images to numpy arrays for pixel-level manipulation
            generated_image_np = np.array(generated_image)
            input_image_np = np.array(input_image)

            # Combine the original image with the inpainted image using the mask
            postprocess_np = np.where(np.expand_dims(mask_image, axis=-1) > 0.5, input_image_np, generated_image_np)

            # Convert back to PIL image
            postprocess_image = Image.fromarray(postprocess_np.astype(np.uint8))
            processed_images.append(postprocess_image)
        else:
            processed_images.append(generated_image)

    # Save the first image with its generation metadata
    
    first_output_image = processed_images[0]
    
    # Create metadata with function parameters
    metadata = PngImagePlugin.PngInfo()
    #metadata.add_text("use_controlnet", str(use_controlnet))
    #metadata.add_text("generator", str(generator))
    metadata.add_text("model/checkpoint", str(checkpoint))
    metadata.add_text("scheduler", str(scheduler))
    metadata.add_text("seed", str(seed))
    metadata.add_text("prompt", prompt)
    metadata.add_text("negative_prompt", negative_prompt)
    metadata.add_text("width", str(width))
    metadata.add_text("height", str(height))
    metadata.add_text("steps", str(steps))
    metadata.add_text("mask_blur", str(mask_blur))
    metadata.add_text("cfg_scale", str(cfg_scale))
    metadata.add_text("clip_skip", str(clip_skip))
    #metadata.add_text("mask_output", str(mask_output))
    metadata.add_text("fill_setting", str(fill_setting))
    #metadata.add_text("segment_type", segment_type)
    metadata.add_text("maintain_aspect_ratio", str(maintain_aspect_ratio))
    metadata.add_text("post_process", str(post_process))
    metadata.add_text("custom_dimensions", str(custom_dimensions))
    metadata.add_text("denoise_strength", str(denoise_strength))
    metadata.add_text("batch_size", str(batch_size))

    # Save the output image with metadata
    first_output_image.save(
    os.path.join("outputs/inpaint", f"output_image_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"), 
    pnginfo=metadata)
    
    first_output_image.save("outputimg1.png", pnginfo=metadata)
        
    return processed_images  # Return as a list of images for Gradio's Gallery    
    

def add_padding(image, target_width, target_height, pad_color=None):
    """Resize and pad image to target dimensions while maintaining aspect ratio."""
    
    # Check if image is empty
    if image is None or image.size == (0, 0):
        raise ValueError("Input image is empty or not valid.")

    img_width, img_height = image.size
    print(f"Original image size: {img_width}x{img_height}")  # Debugging line
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
    print(f"Resized image size (before padding): {image.size}")  # Debugging line
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

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

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



