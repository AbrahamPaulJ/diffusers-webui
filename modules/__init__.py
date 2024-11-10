import os
import diffusers.schedulers 
import gc
import torch

try:
  import google.colab # type: ignore
  IN_COLAB = True
except:
  IN_COLAB = False
  
device = "cuda" if torch.cuda.is_available() else "cpu"
  
is_local = os.getenv("MYAPP_DEV_ENV") == "true"

SCHEDULERS = {
    "DPM++_2M_KARRAS": diffusers.schedulers.DPMSolverMultistepScheduler,  # Add this entry
    "DPM++_2M": diffusers.schedulers.DPMSolverMultistepScheduler,
    "EULER_A": diffusers.schedulers.EulerAncestralDiscreteScheduler,
    "EULER": diffusers.schedulers.EulerDiscreteScheduler,
    "DDIM": diffusers.schedulers.DDIMScheduler,
    "DDPM": diffusers.schedulers.DDPMScheduler,
    "DEIS": diffusers.schedulers.DEISMultistepScheduler,
    "DPM2": diffusers.schedulers.KDPM2DiscreteScheduler,
    "DPM2-A": diffusers.schedulers.KDPM2AncestralDiscreteScheduler,
    "DPM++_2S": diffusers.schedulers.DPMSolverSinglestepScheduler,
    "DPM++_SDE": diffusers.schedulers.DPMSolverSDEScheduler,
    "DPM++_SDE_KARRAS": diffusers.schedulers.DPMSolverSDEScheduler,  # Add this entry
    "UNIPC": diffusers.schedulers.UniPCMultistepScheduler,
    "HEUN": diffusers.schedulers.HeunDiscreteScheduler,
    "HEUN_KARRAS": diffusers.schedulers.HeunDiscreteScheduler,
    "LMS": diffusers.schedulers.LMSDiscreteScheduler,
    "LMS_KARRAS": diffusers.schedulers.LMSDiscreteScheduler,
    "PNDM": diffusers.schedulers.PNDMScheduler,
}

loaded_embeddings = set()


import os

def load_embeddings_for_prompt(pipe, prompt, negative_prompt=None):
    embeds_folder = "embeddings"
    embeddings_found = False  # Flag to check if any embeddings were found
    
    # Combine the positive and negative prompts into one string
    combined_prompt = prompt
    if negative_prompt:
        combined_prompt += " " + negative_prompt
    
    # Loop through each file in the embeds_folder
    for filename in os.listdir(embeds_folder):
        # Check if the file ends with .pt or .safetensors
        if filename.endswith(".pt") or filename.endswith(".safetensors"):
            embeddings_found = True  # Set flag to True if embeddings are found
            
            # Define the full path to the file
            file_path = os.path.join(embeds_folder, filename)
            
            # Get token name by removing the file extension
            token_name = os.path.splitext(filename)[0]
            
            # Check if the token is in the combined prompt and if it hasn't been loaded already
            if token_name in combined_prompt and token_name not in loaded_embeddings:
                # Load the embedding
                pipe.load_textual_inversion(file_path, token=token_name)
                loaded_embeddings.add(token_name)  # Cache the loaded embedding
                print(f"Loaded embedding: {token_name}")
    
    # Debug statement if no embeddings were found
    if not embeddings_found:
        print("No embeddings found in the 'embeddings' folder.")
          
def flush():
    gc.collect()
    if device == "cuda":
      torch.cuda.empty_cache()