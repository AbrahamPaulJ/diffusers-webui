from torch import Generator
from pipelines import PipelineManager

def generate_txt2img_image(pipeline_manager: PipelineManager, selected_model, prompt, negative_prompt, width, height, seed, steps, cfg_scale, clip_skip):
    """Generate an image from text prompt using the loaded pipeline."""

    pipe = pipeline_manager.active_pipe  # Use the active pipeline
    if pipe is None:
        raise ValueError("txt2img pipeline not initialized.")

    # Enforce minimum CFG scale value
    if cfg_scale < 0:
        cfg_scale = 0  # Set to 0 if it's below

    # Convert seed input to an integer, defaulting to -1 for empty input
    if seed is None or seed == "":
        seed = -1  # Default to -1 if empty

    # Check if seed is -1, indicating randomness
    if seed == -1:
        generator = None  # Use a random generator
    else:
        generator = Generator().manual_seed(seed)  # Set the seed for reproducibility

    # Generate the image
    image = pipe(prompt, negative_prompt=negative_prompt, generator=generator, width=width, height=height,
                 num_inference_steps=steps, guidance_scale=cfg_scale, clip_skip=clip_skip).images[0]
    
    return image


