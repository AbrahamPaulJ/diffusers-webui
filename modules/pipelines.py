import diffusers.schedulers
from diffusers import (
    StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline,
    StableDiffusionPipeline, ControlNetModel
)
import torch
import os
import traceback
import importlib.util
from modules import IN_COLAB

if IN_COLAB:
    print("Colab environment detected: Safety checker enabled.")
    
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
has_xformers = importlib.util.find_spec("xformers") is not None
print(f"xformers available: {has_xformers}")

class PipelineManager:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.active_pipe = None
        self.active_checkpoint = None
        self.active_scheduler = None
        self.controlnet_model = None
        self.active_controlnet = None  # To track ControlNet type

    def load_pipeline(self, checkpoint_name: str = "stablediffusionapi/realistic-vision-v6.0-b1-inpaint", 
                      pipeline_type: str = "inpainting", scheduler: str = "DPMSolverMultistepScheduler", 
                      controlnet_type: str = "None"):
        """Load or update the pipeline as needed, handling model, scheduler and ControlNet adjustments."""
        
    
        # Handle None case for self.control_net_type to avoid AttributeError
        current_control_net_type = self.active_controlnet if self.active_controlnet is not None else "None"

        controlnet_changed = (controlnet_type == "None" and current_control_net_type != "None") or \
                                (controlnet_type != "None" and current_control_net_type == "None")                            

        # Reload if a new checkpoint or ControlNet type is specified
        if (checkpoint_name != self.active_checkpoint) or controlnet_changed:
            if self.active_pipe is not None:
                del self.active_pipe
                torch.cuda.empty_cache() if device == "cuda" else None
                print(f"Previous checkpoint {self.active_checkpoint} unloaded.")

            try:
                # Define pipeline class based on type and ControlNet usage
                pipeline_classes = {
                    ('inpainting', True): StableDiffusionControlNetInpaintPipeline,
                    ('inpainting', False): StableDiffusionInpaintPipeline,
                    ('txt2img', False): StableDiffusionPipeline,
                }
                
                # Determine if ControlNet is used
                is_controlnet = controlnet_type != "None"
                pipeline_class = pipeline_classes.get((pipeline_type, is_controlnet))
                if not pipeline_class:
                    raise ValueError("Invalid pipeline type specified.")

                # Load the pipeline from the model or checkpoint
                load_method = pipeline_class.from_single_file if checkpoint_name.endswith(".ckpt") else pipeline_class.from_pretrained
                print(f"Loading checkpoint from: {checkpoint_name}")
                
                pipe = load_method(
                    os.path.join(self.model_dir, checkpoint_name) if checkpoint_name.endswith(".ckpt") else checkpoint_name,
                    controlnet=self.controlnet_model if is_controlnet else None,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=IN_COLAB,  #True in colab env
                    cache_dir=self.model_dir
                )
                
                # Move the pipeline to GPU if available.
                self.active_pipe = pipe.to(device)
                print(f"Using device: {device}")
                self.active_checkpoint = checkpoint_name

                print(f"Loaded checkpoint: {checkpoint_name} (ControlNet Type: {controlnet_type}, Scheduler: {scheduler})")

            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_name}: {e}")
                traceback.print_exc()
                self.active_pipe = None

        # Update ControlNet dynamically
        self.update_controlnet(controlnet_type)

        # Update Scheduler dynamically
        self.update_scheduler(scheduler)
        
        if device == "cuda":
            self.active_pipe.enable_model_cpu_offload()
            if has_xformers:
                self.active_pipe.enable_xformers_memory_efficient_attention()

            current_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"Current GPU usage: {current_memory_allocated:.2f} MB")

    def update_controlnet(self, controlnet_type: str):
        """Enable or disable ControlNet dynamically based on the dropdown selection."""
        if controlnet_type == "None":
            self.controlnet_model = None
            self.active_controlnet = None
            print("ControlNet disabled.")
        else:
            controlnet_model_map = {                
                "Canny - lllyasviel/control_v11p_sd15_canny": "lllyasviel/control_v11p_sd15_canny",
                "Depth - lllyasviel/control_v11f1p_sd15_depth": "lllyasviel/control_v11f1p_sd15_depth",
                "OpenPose - lllyasviel/control_v11p_sd15_openpose": "lllyasviel/control_v11p_sd15_openpose"
            }
            model_name = controlnet_model_map.get(controlnet_type)
            if model_name and model_name != self.active_controlnet:
                try:
                    print(f"Loading ControlNet model: {controlnet_type}")
                    self.controlnet_model = ControlNetModel.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        cache_dir=self.model_dir
                    )
                    self.active_controlnet = model_name
                    print(f"ControlNet model {controlnet_type} loaded.")
                except Exception as e:
                    print(f"Error loading ControlNet {controlnet_type}: {e}")
                    traceback.print_exc()
                    self.controlnet_model = None

        # Update pipeline's ControlNet setting
        if self.active_pipe:
            self.active_pipe.controlnet = self.controlnet_model
            
    def update_scheduler(self, scheduler: str):
        """Change the scheduler dynamically without reloading the pipeline."""
        # Check if the scheduler is different from the active one
        if scheduler != self.active_scheduler:
            try:
                scheduler_class = getattr(diffusers.schedulers, scheduler)

                # Load the scheduler based on checkpoint type
                if not self.active_checkpoint.endswith(".ckpt"):
                    # Use pretrained scheduler for regular models
                    self.active_pipe.scheduler = scheduler_class.from_pretrained(
                        self.active_checkpoint,
                        subfolder="scheduler",
                        cache_dir=self.model_dir
                    )
                else:
                    # For .ckpt models, initialize scheduler from config
                    self.active_pipe.scheduler = scheduler_class.from_config(self.active_pipe.scheduler.config)

                # Update the active scheduler to avoid redundant reloads
                self.active_scheduler = scheduler
                print(f"Scheduler: {scheduler}")

            except Exception as e:
                print(f"Error setting scheduler {scheduler}: {e}")
                traceback.print_exc()