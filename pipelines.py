# pipelines.py

import diffusers.schedulers
from diffusers import (
    StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline,
    StableDiffusionPipeline, ControlNetModel
)
import torch
import os
import traceback

class PipelineManager:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.active_pipe = None
        self.active_checkpoint = None
        self.active_scheduler = None
        self.controlnet_model = None  # Store ControlNet model separately
        self.control_net_enabled = False

    def load_pipeline(self, checkpoint_name: str = "stablediffusionapi/realistic-vision-v6.0-b1-inpaint", 
                      pipeline_type: str = "inpainting", scheduler: str = "DPMSolverMultistepScheduler", 
                      use_controlnet: bool = False):
        """Load or update the pipeline as needed, handling model, scheduler, and ControlNet adjustments."""

        # Reload only if a new checkpoint is specified
        if checkpoint_name != self.active_checkpoint:
            # Unload the existing pipeline to free memory if a new model is loaded
            if self.active_pipe is not None:
                del self.active_pipe
                torch.cuda.empty_cache()
                print(f"Previous checkpoint {self.active_checkpoint} unloaded.")
            
            try:
                # Define pipeline class based on type and ControlNet usage
                pipeline_classes = {
                    ('inpainting', True): StableDiffusionControlNetInpaintPipeline,
                    ('inpainting', False): StableDiffusionInpaintPipeline,
                    ('txt2img', False): StableDiffusionPipeline,
                }
                
                pipeline_class = pipeline_classes.get((pipeline_type, use_controlnet))
                if not pipeline_class:
                    raise ValueError("Invalid pipeline type specified.")
                
                # Load the pipeline from the model or checkpoint
                load_method = pipeline_class.from_single_file if checkpoint_name.endswith(".ckpt") else pipeline_class.from_pretrained
                print(f"Loading checkpoint from: {checkpoint_name}")
                
                pipe = load_method(
                    os.path.join(self.model_dir, checkpoint_name) if checkpoint_name.endswith(".ckpt") else checkpoint_name,
                    controlnet=self.controlnet_model if use_controlnet else None,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                    cache_dir=self.model_dir
                )
                
                # Move the pipeline to GPU and activate it
                self.active_pipe = pipe.to("cuda")
                self.active_checkpoint = checkpoint_name
                self.active_pipe.enable_model_cpu_offload()
                self.active_pipe.enable_xformers_memory_efficient_attention()

                print(f"Loaded checkpoint: {checkpoint_name} (ControlNet Used: {use_controlnet})")

            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_name}: {e}")
                traceback.print_exc()
                self.active_pipe = None

        # Update ControlNet dynamically
        self.update_controlnet(use_controlnet)

        # Update Scheduler dynamically
        self.update_scheduler(scheduler)

        current_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Current GPU usage: {current_memory_allocated:.2f} MB")

    def update_controlnet(self, use_controlnet: bool):
        """Enable or disable ControlNet without reloading the pipeline."""
        if use_controlnet and not self.controlnet_model:
            print("Loading ControlNet model...")
            try:
                # Load ControlNet if it hasn't been loaded before
                self.controlnet_model = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_inpaint", 
                    torch_dtype=torch.float16, 
                    cache_dir=self.model_dir
                )
                print("ControlNet model loaded.")
            except Exception as e:
                print(f"Error loading ControlNet: {e}")
                traceback.print_exc()
                self.controlnet_model = None

        # Toggle ControlNet usage in the pipeline
        self.active_pipe.controlnet = self.controlnet_model if use_controlnet else None
        self.control_net_enabled = use_controlnet
        print(f"ControlNet enabled: {self.control_net_enabled}")

    def update_scheduler(self, scheduler: str):
        """Change the scheduler dynamically without reloading the pipeline."""
        if scheduler != self.active_scheduler:
            try:
                # Dynamically load the specified scheduler
                scheduler_class = getattr(diffusers.schedulers, scheduler)
                if not self.active_checkpoint.endswith(".ckpt"):
                    self.active_pipe.scheduler = scheduler_class.from_pretrained(
                        self.active_checkpoint,
                        subfolder="scheduler",
                        cache_dir=self.model_dir
                    )
                else:
                    # For .ckpt models, initialize scheduler from config
                    self.active_pipe.scheduler = scheduler_class.from_config(self.active_pipe.scheduler.config)
                    
                self.active_scheduler = scheduler
                print(f"Scheduler updated to {scheduler}")
            except Exception as e:
                print(f"Error setting scheduler {scheduler}: {e}")
                traceback.print_exc()
