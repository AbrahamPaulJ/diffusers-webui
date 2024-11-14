from diffusers import (
     StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, 
    StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
)
from compel import Compel, DiffusersTextualInversionManager

import torch
import os
import traceback
from modules import IN_COLAB, SCHEDULERS, DEVICE, VRAM, HAS_XFORMERS, TORCH_DTYPE, flush

if IN_COLAB:
    print("Colab environment detected: Safety checker enabled.")
    
class PipelineManager:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.active_pipe = None
        self.active_checkpoint = None
        self.active_scheduler = None
        self.controlnet_model = None
        self.active_controlnet = None  # To track ControlNet type
        self.active_lora_dict = None
        self.active_pipeline_type = None  # Track active pipeline type
        self.compel=None
        
    def load_pipeline(self, checkpoint_name: str = "stablediffusionapi/realistic-vision-v6.0-b1-inpaint", 
                      pipeline_type: str = "img2img", scheduler: str = "DPM++_2M_KARRAS", 
                      controlnet_type: str = "None", use_lora: bool = False, lora_dict: dict = None):  # New parameter for LoRA
        """Load or update the pipeline as needed, handling model, scheduler, ControlNet, and LoRA adjustments."""
             
        current_control_net_type = self.active_controlnet if self.active_controlnet is not None else "None"

        controlnet_changed = (controlnet_type == "None" and current_control_net_type != "None") or \
                                (controlnet_type != "None" and current_control_net_type == "None")
                                                  
        pipe_reload_condition = (checkpoint_name != self.active_checkpoint) or controlnet_changed or (pipeline_type != self.active_pipeline_type) or (self.active_lora_dict!=lora_dict)
             
        # Reload if a new checkpoint or ControlNet type is specified
        if pipe_reload_condition:
            if self.active_pipe is not None:
                del self.active_pipe
                flush()
                print(f"Previous checkpoint {self.active_checkpoint} unloaded.")

            try:
                # Define pipeline class based on type and ControlNet usage
                pipeline_classes = {
                    ('img2img', True): StableDiffusionControlNetImg2ImgPipeline,
                    ('img2img', False): StableDiffusionImg2ImgPipeline,
                    ('inpainting', True): StableDiffusionControlNetInpaintPipeline,
                    ('inpainting', False): StableDiffusionInpaintPipeline,
                    ('txt2img', True): StableDiffusionControlNetPipeline,
                    ('txt2img', False): StableDiffusionPipeline,
                }
                
                # Determine if ControlNet is used
                is_controlnet = controlnet_type != "None"
                pipeline_class = pipeline_classes.get((pipeline_type, is_controlnet))
                
                if not pipeline_class:
                    raise ValueError("Invalid pipeline type specified.")

                # Load the pipeline from the model or checkpoint
                load_method = pipeline_class.from_single_file if checkpoint_name.endswith((".ckpt", ".safetensors")) else pipeline_class.from_pretrained
                print(f"Loading checkpoint from: {checkpoint_name}")
                
                pipe = load_method(
                    os.path.join(self.model_dir, checkpoint_name) if checkpoint_name.endswith((".ckpt", ".safetensors")) else checkpoint_name,
                    torch_dtype=TORCH_DTYPE,
                    controlnet= self.controlnet_model,
                    safety_checker=None,
                    requires_safety_checker=IN_COLAB,  # True in colab env
                    cache_dir=self.model_dir
                )
                textual_inversion_manager = DiffusersTextualInversionManager(pipe)
                self.compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,textual_inversion_manager=textual_inversion_manager, truncate_long_prompts=False)
                                
                # Move the pipeline to GPU if available.
                self.active_pipe = pipe
                if VRAM < 7:
                    print(f"VAE tiling enabled.")
                    self.active_pipe.enable_vae_tiling()
                
                if DEVICE == "cuda" and HAS_XFORMERS:
                    self.active_pipe.enable_xformers_memory_efficient_attention()
                          
                self.active_checkpoint = checkpoint_name
                self.active_pipeline_type = pipeline_type  # Update active pipeline type
                
                if DEVICE == "cuda":
                    self.active_pipe.enable_model_cpu_offload()
                    
                    current_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    print(f"Current GPU usage: {current_memory_allocated:.2f} MB")
                
                # print(f"PRC:{pipe_reload_condition}")
                # print(f"CnC:{controlnet_changed}")
                # print(f"PTC{(pipeline_type != self.active_pipeline_type)}")
                # print(f"LC:{(self.active_lora_dict!=lora_dict)}")
                # print(f"CkC:{(checkpoint_name != self.active_checkpoint)}")
                print(f"Loaded checkpoint: {checkpoint_name} (ControlNet Type: {controlnet_type}, Scheduler: {scheduler}, LoRAs: {'None' if not use_lora else lora_dict})")

            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_name}: {e}")
                traceback.print_exc()
                self.active_pipe = None

        # Update ControlNet dynamically
        self.update_controlnet(controlnet_type)

        # Update Scheduler dynamically
        self.update_scheduler(scheduler, pipe_reload_condition)
        
        self.update_lora(use_lora, lora_dict)
            
    def update_controlnet(self, controlnet_type: str):
        """Enable or disable ControlNet dynamically based on the dropdown selection."""
        if controlnet_type == "None":
            self.controlnet_model = None    
            if self.active_controlnet!=None:
                print("ControlNet disabled.") 
            self.active_controlnet = None
            
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
                        torch_dtype=TORCH_DTYPE,
                        cache_dir=self.model_dirs
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
            
    def update_scheduler(self, scheduler: str, pipe_reload_condition):
        """Change the scheduler dynamically without reloading the pipeline."""
        # Check if the scheduler is different from the active one or if controlnet has changed
        if scheduler != self.active_scheduler or pipe_reload_condition:
            try:
                # Retrieve the scheduler class from the SCHEDULERS dictionary using the short name
                scheduler_class = SCHEDULERS.get(scheduler)
                if not scheduler_class:
                    raise ValueError(f"Scheduler '{scheduler}' not found in SCHEDULERS dictionary.")
                
                # Load the scheduler based on checkpoint type
                if not self.active_checkpoint.endswith((".ckpt", ".safetensors")):
                    # Use pretrained scheduler for regular models
                    self.active_pipe.scheduler = scheduler_class.from_pretrained(
                        self.active_checkpoint,
                        subfolder="scheduler",
                        cache_dir=self.model_dir
                    )
                else:
                    # For ckpt or safetensor models, initialize scheduler from config
                    self.active_pipe.scheduler = scheduler_class.from_config(self.active_pipe.scheduler.config)
                
                # Update the active scheduler to avoid redundant reloads
                self.active_scheduler = scheduler
                print(f"Scheduler updated to: {scheduler}")

            except Exception as e:
                print(f"Error setting scheduler {scheduler}: {e}")
                traceback.print_exc()


    def update_lora(self, use_lora, lora_dict, lora_adapter_names=[], lora_scales=[], fuse_scale=1.0):
        """Update the LoRA weights with scaling factors for multiple LoRAs."""
        
        # Construct lora_dirs from lora_names
        
        if (use_lora and (lora_dict!= self.active_lora_dict)): 
            self.active_pipe.unload_lora_weights()
            lora_paths = list(lora_dict.keys())
            lora_scales = list(lora_dict.values())
            # Ensure there are enough adapter names for the LoRAs
            if len(lora_adapter_names) < len(lora_paths):
                # Create adapter names based on filenames
                adapters = [os.path.splitext(os.path.basename(p))[0] for p in lora_paths[len(lora_adapter_names):]]
                adapters = [a.replace(".", "_") for a in adapters]  # Replace '.' with '_' for safe variable names
                lora_adapter_names += adapters
            # Debug: Print the lora_adapter_names after ensuring they are set correctly
            print("lora_adapter_names:", lora_adapter_names)

            # Ensure there are enough scales for the LoRAs
            if len(lora_scales) < len(lora_paths):
                lora_scales += [1.0] * (len(lora_paths) - len(lora_scales))

            # Debug: Print the lora_scales to check if scales are set correctly
            print("lora_scales:", lora_scales)

            # Load each LoRA weight and associate with the correct adapter name
            for lp, la in zip(lora_paths, lora_adapter_names):
                try:
                    print(f"Loading LoRA weights from: {lp} with adapter name: {la}")
                    self.active_pipe.load_lora_weights(lp, adapter_name=la)  # Load LoRA with the adapter name
                except Exception as e:
                    print(f"Error loading LoRA weight {lp}: {e}")

            # Set the adapters and their respective scales
            try:
                print("Setting adapters...")
                self.active_pipe.set_adapters(lora_adapter_names, adapter_weights=lora_scales)

                # Fuse the LoRAs with the given scale
                print(f"Fusing LoRAs with scale {fuse_scale}...")
                self.active_pipe.fuse_lora(adapter_names=lora_adapter_names, lora_scale=fuse_scale)
                self.active_lora_dict = lora_dict
            except Exception as e:
                print(f"Error setting adapters or fusing LoRAs: {e}")

            print("LoRAs loaded and fused.")  
            

        else:
            if not use_lora:
            # Unload LoRA weights if not using LoRA
                if self.active_pipe is not None:
                    self.active_pipe.unload_lora_weights()
                    # print("No LoRA weights.")
                self.active_lora_dict = None  # Reset active LoRA if none is provided

