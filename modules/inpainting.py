# modules/inpainting.py

from modules import is_local, load_embeddings_for_prompt
from modules.pipelines import PipelineManager
from modules.image_processing import *

from datetime import datetime
import os
import time
from PIL import Image, PngImagePlugin, ImageFilter
import numpy as np

def time_execution(fn):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        print(f"{fn.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@time_execution 
def generate_inpaint_image(pipeline_manager: PipelineManager, controlnet_name, seed, generator, prompt, negative_prompt, width, height, steps, cfg_scale, clip_skip, inpaint_mask, fill_setting, input_image, maintain_aspect_ratio, post_process, denoise_strength, batch_size, mask_blur, mode, outpaint_img_pos, outpaint_max_dim, controlnet_strength, use_lora, lora_dropdown, lora_prompt):
    """Generate an inpainting image using the loaded pipeline."""
    
    pipe = pipeline_manager.active_pipe
    
    if pipe is None:
        raise ValueError("Inpainting pipeline not initialized.")
    
    compel = pipeline_manager.compel
    
    # Cache to track already loaded embeddings
    load_embeddings_for_prompt(pipe, prompt, negative_prompt=negative_prompt)
    
    conditioning = compel.build_conditioning_tensor(prompt)
    negative_conditioning = compel.build_conditioning_tensor(negative_prompt)
    [con_embeds, neg_embeds] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
    
     
    if mode=="Inpaint":
        width, height = int(width), int(height)
        # if not custom_dimensions:
        #     width, height = auto_select_dimensions(input_image)
        #     print(f"Target dimensions are {height}x{width}")
                 
        full_mask = inpaint_mask
        inpaint_mask = use_brush(inpaint_mask)
        inpaint_mask = inpaint_mask.resize((input_image.width, input_image.height), Image.Resampling.LANCZOS) 
        
        # Convert the mask image to grayscale
        inpaint_mask = inpaint_mask.convert("L")

        if maintain_aspect_ratio:
            inpaint_mask = add_padding(inpaint_mask, width, height, pad_color=0)  # Black padding for the mask
            input_image = add_padding(input_image, width, height)  # Use average color for input image
        
        if mask_blur > 0:    
            inpaint_mask = inpaint_mask.filter(ImageFilter.GaussianBlur(mask_blur))
            
        mask_array = np.array(inpaint_mask) / 255.0

        mask_image = mask_array if fill_setting == "Inpaint Masked" else 1.0 - mask_array
        if is_local:
            mask_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
            mask_pil.save("inpaintmaskimg.png") 
        
        # print("Input image size (PIL):", input_image.size)  # Width, Height
        # print("Input image color mode (PIL):", input_image.mode)  # e.g., RGB, RGBA, L
        # print("Mask image shape (NumPy):", mask_image.shape)    # e.g., (height, width, channels or none if grayscale)
        
    if mode=="Outpaint":
         
        outpaint_mask = use_crop(inpaint_mask)
        outpaint_max_dim = int(outpaint_max_dim) 
        outpaint_mask = resize_to_max_side(outpaint_mask, target_size = outpaint_max_dim)  
        inner_dimensions = get_inner_dimensions(outpaint_mask)
        bounding_box_sides = get_bounding_box_sides(outpaint_mask)
        left, right, top, bottom = bounding_box_sides
        padded_width = inner_dimensions[0] + left + right
        padded_height = inner_dimensions[1] + top + bottom
        resized_input_image = input_image.resize(inner_dimensions, Image.Resampling.LANCZOS)
        
        # Create a new blank image with padding dimensions and a black background in RGB
        padded_image = Image.new("RGB", (padded_width, padded_height), (0, 0, 0))  # Black background
        padded_image.paste(resized_input_image.convert('RGB'), (left, top))  # Ensure resized_input_image is in RGB
        # `padded_image` now has the resized image centered with black padding on all sides
        padded_image = resize_to_nearest_multiple_of_8(padded_image)
        if is_local:
            padded_image.save("outpaintinputimg.png")
        width, height = padded_image.size
        mask_image = Image.new("RGBA", (resized_input_image.width, resized_input_image.height), (255, 255, 255, 255))
        padded_mask_image = Image.new("RGBA", (padded_width, padded_height), (0, 0, 0, 255))  # Black background
        padded_mask_image.paste(mask_image, (left, top))
        padded_mask_image = padded_mask_image.resize((padded_image.width, padded_image.height), Image.Resampling.LANCZOS).convert("L")

        if mask_blur > 0:    
            padded_mask_image = padded_mask_image.filter(ImageFilter.GaussianBlur(mask_blur))
        if is_local:
            padded_mask_image.save("outpaintmaskimg.png")
        mask_image = np.array(padded_mask_image) / 255     
        input_image=padded_image
        # print("Input image size (PIL):", input_image.size)  # Width, Height
        # print("Input image color mode (PIL):", input_image.mode)  # e.g., RGB, RGBA, L
        # print("Mask image shape (NumPy):", mask_image.shape)    # e.g., (height, width, channels or none if grayscale)
    
    # Prepare the keyword arguments for the pipeline
    pipe_kwargs = {
        "num_images_per_prompt": batch_size,
        # "prompt": prompt,
        # "negative_prompt": negative_prompt,
        "prompt_embeds":con_embeds,
        "negative_prompt_embeds":neg_embeds,
        "generator": generator,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "strength": denoise_strength,
        "guidance_scale": cfg_scale,
        "clip_skip": clip_skip,
        "mask_image": 1.0 - mask_image, 
        "image": input_image,
    }
    

    # Add control_image to the keyword arguments if ControlNet is used
    controlnet = pipeline_manager.active_controlnet
    
    if controlnet != None:
        control_image = create_control_image(input_image, controlnet)
        if is_local:
            control_image.save("controlimg.png")
        controlnet_strength = controlnet_strength
        pipe_kwargs["control_image"] = control_image
        pipe_kwargs["controlnet_strength"] = controlnet_strength       
      
    # Generate the image with the prepared arguments
    generated_images = pipe(**pipe_kwargs).images
    
    if mode=="Inpaint":
        if not maintain_aspect_ratio and post_process:
            input_image = input_image.resize(generated_images[0].size, Image.Resampling.LANCZOS)
            mask_image = Image.fromarray(mask_image)
            mask_image = mask_image.resize(generated_images[0].size, Image.Resampling.LANCZOS)
            mask_image = np.array(mask_image)/255.0
            
    processed_images = []
    input_image_np = np.array(input_image)

    for generated_image in generated_images:
            if post_process:
                    
                generated_image_np = np.array(generated_image)

                # Combine the original image with the inpainted image using the mask
                postprocess_np = np.where(np.expand_dims(mask_image, axis=-1) > 0.5, input_image_np, generated_image_np)

                # Convert back to PIL image
                postprocess_image = Image.fromarray(postprocess_np.astype(np.uint8))
                processed_images.append(postprocess_image)
            else:
                processed_images.append(generated_image)


    # Save the first image with its generation metadata
    
    first_output_image = processed_images[0]
    
    if mode == "Inpaint":
        full_mask = full_mask["layers"][0]
        output_directory = os.path.join("outputs", "inpaint_masks")  # Cross-platform path
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)
        # Save the first image with its generation metadata
        mask_name = f"mask_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        mask_path = os.path.join(output_directory, mask_name)
        full_mask.save(mask_path)

    directory = "inpaint" if mode=="Inpaint" else "outpaint"
    output_directory = os.path.join("outputs", directory) 
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    output_name = f"output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    output_path = os.path.join(output_directory, output_name)
        
    # Create metadata with function parameters
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("pipeline", type(pipeline_manager.active_pipe).__name__)
    metadata.add_text("model/checkpoint", str(pipeline_manager.active_checkpoint))
    metadata.add_text("scheduler", str(pipeline_manager.active_scheduler))
    metadata.add_text("controlnet", str(controlnet_name))
    metadata.add_text("controlnet_strength", str(controlnet_strength))
    metadata.add_text("seed", str(seed))
    metadata.add_text("prompt", prompt)
    metadata.add_text("negative_prompt", negative_prompt)
    metadata.add_text("width", str(width))
    metadata.add_text("height", str(height))
    metadata.add_text("steps", str(steps))
    metadata.add_text("mask_blur", str(mask_blur))
    metadata.add_text("cfg_scale", str(cfg_scale))
    metadata.add_text("clip_skip", str(clip_skip))
    metadata.add_text("mode", str(mode))
    if mode=="Inpaint":
        metadata.add_text("output_path", output_path)
        metadata.add_text("mask_path", mask_path)
        metadata.add_text("fill_setting", str(fill_setting))
        metadata.add_text("maintain_aspect_ratio", str(maintain_aspect_ratio))
    metadata.add_text("post_process", str(post_process))
    metadata.add_text("denoise_strength", str(denoise_strength))
    metadata.add_text("batch_size", str(batch_size))
    if mode=="Outpaint":
        metadata.add_text("image_positioned_at", str(outpaint_img_pos))
        metadata.add_text("maximum_width/height", str(outpaint_max_dim))  
    if use_lora:
        metadata.add_text("use_lora", str(use_lora))
        metadata.add_text("loras_used", ",".join(lora_dropdown)) 
        metadata.add_text("lora_weights",str(lora_prompt))  
        
    first_output_image.save(output_path, pnginfo=metadata)
    
    if is_local:
        first_output_image.save("outputimg1.png")
        
    return processed_images  # Return as a list of images for Gradio's Gallery    
    







