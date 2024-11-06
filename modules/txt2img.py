# modules/txt2img.py

from modules import is_local
from modules.pipelines import PipelineManager
from modules.image_processes import *

from datetime import datetime
import os
import time
from PIL import PngImagePlugin

def time_execution(fn):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        print(f"{fn.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@time_execution 
def generate_txt2img_image(pipeline_manager: PipelineManager, controlnet_name, seed, generator, prompt, negative_prompt, width, height, steps, cfg_scale, clip_skip, control_input, denoise_strength, batch_size, controlnet_strength):
    """Generate an image from text prompt using the loaded pipeline."""

    pipe = pipeline_manager.active_pipe
    
    if pipe is None:
        raise ValueError("Inpainting pipeline not initialized.")

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
    "clip_skip": clip_skip
}
    
    # Add control_image to the keyword arguments if ControlNet is used
    controlnet = pipeline_manager.active_controlnet
    
    if controlnet != None:
        control_image = create_control_image(control_input, controlnet)
        if is_local:
            control_image.save("controlimg.png")
        controlnet_strength = controlnet_strength
        pipe_kwargs["control_image"] = control_image
        pipe_kwargs["controlnet_strength"] = controlnet_strength
      
    # Print types for debugging
    for key, value in pipe_kwargs.items():
        print(f"{key}: Type = {type(value)}, Value = {value}")

    generated_images = pipe(**pipe_kwargs).images
    
    first_output_image = generated_images[0]
    
    output_directory = os.path.join("outputs", "txt2img") 
    os.makedirs(output_directory, exist_ok=True)
    output_name = f"output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    output_path = os.path.join(output_directory, output_name)
    
    # Create metadata with function parameters
    metadata = PngImagePlugin.PngInfo()
    #metadata.add_text("generator", str(generator))
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
    metadata.add_text("cfg_scale", str(cfg_scale))
    metadata.add_text("clip_skip", str(clip_skip))
    metadata.add_text("output_path", output_path)
    metadata.add_text("denoise_strength", str(denoise_strength))
    metadata.add_text("batch_size", str(batch_size))

    # Save the first image with its generation metadata
    first_output_image.save(output_path, pnginfo=metadata)
    
    if is_local:
        first_output_image.save("outputtxt2img.png")

    return generated_images


