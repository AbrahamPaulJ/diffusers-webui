import diffusers.schedulers
from diffusers import (StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline, 
                       StableDiffusionPipeline, ControlNetModel)
import torch
import os
import traceback

class PipelineManager:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.active_pipe = None  # Store only the active pipeline
        self.active_checkpoint = None
        self.active_scheduler = None
        self.control_net_enabled = None

    def load_pipeline(self, checkpoint_name: str = "stablediffusionapi/realistic-vision-v6.0-b1-inpaint", 
                    pipeline_type: str = "inpainting",scheduler: str = "DPMSolverMultistepScheduler", use_controlnet: bool = False):
        """Load the specified pipeline, handling memory and ControlNet as needed."""
        
        # Skip reloading if the requested checkpoint and settings are the same
        if checkpoint_name == self.active_checkpoint and scheduler == self.active_scheduler and self.control_net_enabled == use_controlnet:
            print(f"Checkpoint {checkpoint_name} is already active, no change in other params.")
            return

        # Unload the existing pipeline to free memory
        if self.active_pipe is not None:
            del self.active_pipe
            torch.cuda.empty_cache()
            print(f"Previous checkpoint {self.active_checkpoint} unloaded.")

        try:          
            # Load ControlNet if required
            controlnet = None
            if use_controlnet:
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_inpaint", 
                    torch_dtype=torch.float16, 
                    cache_dir=self.model_dir
                )
                self.control_net_enabled = True
            else:
                self.control_net_enabled = False

            # Define pipeline class based on type and ControlNet usage
            pipeline_classes = {
                ('inpainting', True): StableDiffusionControlNetInpaintPipeline,
                ('inpainting', False): StableDiffusionInpaintPipeline,
                ('txt2img', False): StableDiffusionPipeline,
            }

            pipeline_class = pipeline_classes.get((pipeline_type, use_controlnet))
            if not pipeline_class:
                raise ValueError("Invalid pipeline type specified.")
            
            load_method = pipeline_class.from_single_file if checkpoint_name.endswith(".ckpt") else pipeline_class.from_pretrained

            print(f"Loading checkpoint from: {checkpoint_name}")
            print(f"Using load method: {load_method}")
            pipe = load_method(
                os.path.join(self.model_dir, checkpoint_name) if checkpoint_name.endswith(".ckpt") else checkpoint_name ,
                controlnet=controlnet if use_controlnet else None,
                torch_dtype=torch.float16,
                safety_checker=None,
                local_files_only =True,
                requires_safety_checker=False,
                cache_dir=self.model_dir
            )

            # Move the pipeline to the GPU and activate it
            self.active_pipe = pipe.to("cuda")
            self.active_checkpoint = checkpoint_name
            
            self.active_scheduler = scheduler
            scheduler_class = getattr(diffusers.schedulers, scheduler)

            if not checkpoint_name.endswith(".ckpt"):
                pipe.scheduler = scheduler_class.from_pretrained(
                    checkpoint_name,
                    subfolder="scheduler",
                    cache_dir=self.model_dir)
            else:
                pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config) 
                
            
                
            # Enable memory-efficient attention
            pipe.enable_xformers_memory_efficient_attention()

            print(f"Loaded checkpoint: {checkpoint_name} with scheduler {scheduler} (ControlNet Used: {use_controlnet})")
            current_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"Current GPU usage: {current_memory_allocated:.2f} MB")
            

        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_name}: {e}")
            traceback.print_exc() 
            self.active_pipe = None
            



