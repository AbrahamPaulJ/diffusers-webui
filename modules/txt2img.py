# modules/txt2img.py

from modules import IS_LOCAL, DEVICE, HAS_XFORMERS, load_embeddings_for_prompt, flush, loaded_embeddings
from modules.pipelines import PipelineManager
from modules.image_processing import create_control_image


from datetime import datetime
import os
import time
from PIL import PngImagePlugin, Image

def time_execution(fn):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        print(f"{fn.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@time_execution 
def generate_t2i_image(pipeline_manager: PipelineManager, base_model, controlnet_name, seed, generator, prompt, negative_prompt, width, height, steps, cfg_scale, clip_skip, control_input: Image.Image, batch_size, controlnet_strength: float, hires_fix, use_lora, lora_dropdown, lora_prompt):
    """Generate an image from text prompt using the loaded pipeline."""

    pipe = pipeline_manager.active_pipe
    if pipe is None:
        raise ValueError("Inpainting pipeline not initialized.")

    print("Control image color mode (PIL):", control_input.mode)  

    # Cache to track already loaded embeddings
    load_embeddings_for_prompt(pipe, prompt, negative_prompt=negative_prompt)
    print(f"Using embeddings: {loaded_embeddings}")
    
    control_input = control_input
    compel = pipeline_manager.compel
    
    if base_model == 'SD':
        con_embeds = compel.build_conditioning_tensor(prompt)
        neg_embeds = compel.build_conditioning_tensor(negative_prompt)
        [con_embeds, neg_embeds] = compel.pad_conditioning_tensors_to_same_length([con_embeds, neg_embeds])

    elif base_model == 'SDXL':
        # Build positive and negative prompt embeddings
        con_embeds, con_pooled = compel.build_conditioning_tensor(prompt)
        neg_embeds, neg_pooled = compel.build_conditioning_tensor(negative_prompt)
        [con_embeds, neg_embeds] = compel.pad_conditioning_tensors_to_same_length([con_embeds, neg_embeds])
    else:
        print("Invalid base model.")
    

    width, height = int(width), int(height)

    # Prepare kwargs for pipeline
    pipe_kwargs = {
        "num_images_per_prompt": batch_size,
        "prompt_embeds": con_embeds,
        "negative_prompt_embeds": neg_embeds,
        "generator": generator,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "clip_skip": clip_skip
    }

    if base_model == 'SDXL':
        # Add pooled embeddings for SDXL
        pipe_kwargs.update({
            "pooled_prompt_embeds": con_pooled,
            "negative_pooled_prompt_embeds": neg_pooled,
        })


    # Add control_image to the keyword arguments if ControlNet is used
    controlnet = pipeline_manager.active_controlnet
    
    if controlnet != None:
        print("Control image color mode (PIL):", control_input.mode)  
        control_image = create_control_image(control_input, controlnet)
        print("Control image color mode (PIL):", control_image.mode)  
        print(type(control_image))
        if IS_LOCAL:
            control_image.save("controlimg.png")
        pipe_kwargs["image"] = control_image
        # pipe_kwargs["controlnet_conditioning_scale"] = float(controlnet_strength)
      
    # Print types for debugging
    # for key, value in pipe_kwargs.items():
    #     print(f"{key}: Type = {type(value)}, Value = {value}")

    generated_images = pipe(**pipe_kwargs).images
    
    first_output_image = generated_images[0]
    
    output_directory = os.path.join("outputs", "txt2img") 
    os.makedirs(output_directory, exist_ok=True)
    output_name = f"output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    output_path = os.path.join(output_directory, output_name)
    
    # Create metadata with function parameters
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("base_model", str(pipeline_manager.active_base_model))
    metadata.add_text("pipeline", type(pipeline_manager.active_pipe).__name__)
    metadata.add_text("model/checkpoint", str(pipeline_manager.active_checkpoint))
    metadata.add_text("scheduler", str(pipeline_manager.active_scheduler))
    if controlnet_name!="None":
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
    metadata.add_text("batch_size", str(batch_size))
    metadata.add_text("hires_fix_2x", str(hires_fix))
    if use_lora:
        metadata.add_text("use_lora", str(use_lora))
        metadata.add_text("loras_used", ",".join(lora_dropdown)) 
        metadata.add_text("lora_weights",str(lora_prompt))

    if hires_fix: 
        
        from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
        from diffusers.models.attention_processor import AttnProcessor2_0
        from RealESRGAN import RealESRGAN
        from PIL import Image
        
        print("pipe details:")
        print(pipe)
        print(base_model)
        
        if base_model=='SDXL':
            i2ipipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        else:
            i2ipipe = StableDiffusionImg2ImgPipeline(**pipe.components) 

        pipe.enable_vae_tiling()

        pipe.unet.set_attn_processor(AttnProcessor2_0())

        i2ipipe.enable_vae_tiling()
        
        if DEVICE == "cuda" and HAS_XFORMERS:
            i2ipipe.enable_xformers_memory_efficient_attention()
            
        if DEVICE == "cuda":
            i2ipipe.enable_model_cpu_offload()

        i2ipipe.unet.set_attn_processor(AttnProcessor2_0())

        model = RealESRGAN(DEVICE, scale=4)
        model.load_weights('models\RealESRGAN_x4.pth', download=False)
        print("Upscaling...")
        sr_image = model.predict(first_output_image)  
        print("Upscaled")
        
        flush()
        
        scaled_img = sr_image.resize((2*width,2*height), Image.BILINEAR)
        
        i2i_pipe_kwargs = {
            "image": scaled_img,
            "prompt_embeds": con_embeds,
            "negative_prompt_embeds": neg_embeds,
            "generator": generator,
            "width": 2*width,
            "height": 2*height,
            "num_inference_steps": steps,
            "guidance_scale": 1,
            "clip_skip": 2,
            "strength": 0.3
    }
        
        if base_model == 'SDXL':
            # Add pooled embeddings for SDXL
            i2i_pipe_kwargs.update({
                "pooled_prompt_embeds": con_pooled,
                "negative_pooled_prompt_embeds": neg_pooled,
            })
        
        
        print("Applying hires .fix...")
        generated_images = i2ipipe(**i2i_pipe_kwargs).images
        print("hires .fix applied successfully.")
             
        del i2ipipe
        flush()
            
        generated_images[0].save(output_path, pnginfo=metadata)
        
        if IS_LOCAL:
            first_output_image.save("output_t2i.png")
            
        return generated_images
    
    # Save the first image with its generation metadata
    first_output_image.save(output_path, pnginfo=metadata)
    
    if IS_LOCAL:
        first_output_image.save("output_t2i.png")

    return generated_images


