  
import gradio as gr
import os
import random
from modules.image_processing import auto_select_dimensions, remove_background, create_composite, retrieve_mask
from PIL import Image
  
#Functions
def clear_gallery():
    return []

def make_visible():
    return gr.update(visible=True)

def hide():
    return gr.update(visible=False)
                        
def button_is_waiting():
    return gr.update(interactive=False, value="Loading...")

def generating():
    return gr.update(interactive=False, value="Generating...")
 
def toggle_mode(mode):
    if mode=="Inpaint":
        brush = gr.Brush(colors=["#000000"], color_mode='fixed',default_size=50)
        return gr.update(brush=brush, transforms=())
    if mode=="Outpaint":
        return gr.update(brush=False, transforms=('crop'))
    
def toggle_mode_hide(mode,component_hide_count):
    # Determine the visibility state based on the mode
    visible = (mode == "Inpaint")  # True if Inpaint 
    updates = [gr.update(visible=visible) for _ in range(component_hide_count)] 
    return updates  # Return the list of updates

def toggle_mode_show(mode,component_show_count):
    visible = (mode == "Outpaint")  # True if Outpaint
    updates = [gr.update(visible=visible) for _ in range(component_show_count)] 
    return updates  # Return the list of updates

def reset_seed():
    return -1

def random_seed():
    return random.randint(0, 2**32 - 1)

def auto_dim(image):
    if image is not None:
        return auto_select_dimensions(image)
    else:
        return 512,512
    
def update_lora_dropdown():
    loras_folder = "loras"
    # Get the updated list of .safetensors files in the folder
    choices = ["None"] + [file for file in os.listdir(loras_folder) if file.endswith(".safetensors")]
    lora_default_value = choices[0] if choices else None

    # Return updated dropdown values
    return gr.update(choices=choices, value=lora_default_value)

def using_lora(value):
    return gr.update(visible=False) if not value else gr.update(visible=True), gr.update(visible=False) if not value else gr.update(visible=True)

def lora_to_prompt(lora_list):
    lora_prompt_list = []
    
    for lora in lora_list:
        # Remove the ".safetensors" extension
        lora_name = os.path.splitext(lora)[0]
        
        # Format the prompt as <lora:name:scale>
        # Default scale is 1.0 if not specified
        prompt = f"<lora:{lora_name}:1.0>"
        
        # Append the formatted prompt to the list
        lora_prompt_list.append(prompt)
    
    # Return a comma-separated string of the prompts
    return ", ".join(lora_prompt_list)

def lora_to_prompt_cb(checkbox, lora_list):
    if checkbox:
        lora_prompt_list = []
        for lora in lora_list:
            lora_name = os.path.splitext(lora)[0]
            prompt = f"<lora:{lora_name}:1.0>"
            lora_prompt_list.append(prompt)

        return ", ".join(lora_prompt_list)
    
                      
def get_metadata(image):
    if image is None:
        return "No metadata available.", gr.update(visible=False), gr.update(visible=False)
    
    # Extract metadata
    metadata = image.info  # Metadata dictionary for PNG images in PIL
    if not metadata:
        return "No metadata found in this image.", gr.update(visible=False), gr.update(visible=False)
    parameters_string = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    parameters = extract_metadata(parameters_string)
    output_path = parameters.get("output_path")
    if output_path:
        output_path.replace("\\", "/")
    # print(output_path)
    # print(output_path.startswith("outputs\txt2img"))
    txt2img_btn_visible =True if (output_path and output_path.startswith("outputs/txt2img")) else False

    # Format metadata for display
    return "\n".join([f"{k}: {v}" for k, v in metadata.items()]), gr.update(visible=not txt2img_btn_visible), gr.update(visible=txt2img_btn_visible)


def extract_metadata(info):
    # Split the input into lines
    lines = info.strip().split('\n')
    
    # Create a dictionary to hold the parsed parameters
    parameters = {}
    
    # Iterate through each line and extract key-value pairs
    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)  # Split only on the first occurrence
            parameters[key.strip()] = value.strip()
    
    return parameters



def load_info_to_inpaint(info,state):
    # Extract metadata
    parameters = extract_metadata(info)
    background_path = parameters.get("output_path")
    background = Image.open(background_path)
    state^= 1
    
    # Return updates for each component based on the parsed parameters
    return (
        parameters.get("model/checkpoint", "stablediffusionapi/realistic-vision-v6.0-b1-inpaint"),
        parameters.get("scheduler", "DPMSolverMultistepScheduler"),
        parameters.get("controlnet", "None"),
        parameters.get("controlnet_strength", 1.0),
        parameters.get("seed", -1), 
        background,
        parameters.get("prompt", ""),             
        parameters.get("negative_prompt", ""),        
        int(parameters.get("width", 512)),              
        int(parameters.get("height", 768)),            
        int(parameters.get("steps", 25)),   
        float(parameters.get("mask_blur", 0)),       
        float(parameters.get("cfg_scale", 7.0)),     
        int(parameters.get("clip_skip", 1)),             
        parameters.get("fill_setting", "Inpaint Masked"),  
        parameters.get("maintain_aspect_ratio", "True") == "True", 
        parameters.get("post_process", "True") == "True",  
        # parameters.get("custom_dimensions", "True") == "True",  
        float(parameters.get("denoise_strength", 1.0)),  
        int(parameters.get("batch_size", 1)),
        parameters.get("mode", "Inpaint"),
        parameters.get("image_positioned_at", "Center"),
        parameters.get("maximum_width/height", 768),
        state
    )
    
def load_info_to_txt2img(info):
    # Extract metadata
    parameters = extract_metadata(info)
    
    # Debug print statements for parameter values and their types
    print("Debugging Parameters and Types:")
    print(f"Checkpoint: {parameters.get('model/checkpoint', 'stablediffusionapi/realistic-vision-v6.0-b1-inpaint')} (Type: {type(parameters.get('model/checkpoint', 'stablediffusionapi/realistic-vision-v6.0-b1-inpaint'))})")
    print(f"Scheduler: {parameters.get('scheduler', 'DPMSolverMultistepScheduler')} (Type: {type(parameters.get('scheduler', 'DPMSolverMultistepScheduler'))})")
    print(f"ControlNet: {parameters.get('controlnet', 'None')} (Type: {type(parameters.get('controlnet', 'None'))})")
    print(f"ControlNet Strength: {parameters.get('controlnet_strength', 1.0)} (Type: {type(parameters.get('controlnet_strength', 1.0))})")
    print(f"Seed: {parameters.get('seed', -1)} (Type: {type(parameters.get('seed', -1))})")
    print(f"Prompt: {parameters.get('prompt', '')} (Type: {type(parameters.get('prompt', ''))})")
    print(f"Negative Prompt: {parameters.get('negative_prompt', '')} (Type: {type(parameters.get('negative_prompt', ''))})")
    print(f"Width: {parameters.get('width', 512)} (Type: {type(parameters.get('width', 512))})")
    print(f"Height: {parameters.get('height', 768)} (Type: {type(parameters.get('height', 768))})")
    print(f"Steps: {parameters.get('steps', 25)} (Type: {type(parameters.get('steps', 25))})")
    print(f"CFG Scale: {parameters.get('cfg_scale', 7.0)} (Type: {type(parameters.get('cfg_scale', 7.0))})")
    print(f"Clip Skip: {parameters.get('clip_skip', 1)} (Type: {type(parameters.get('clip_skip', 1))})")
    print(f"Batch Size: {parameters.get('batch_size', 1)} (Type: {type(parameters.get('batch_size', 1))})")
    
    # Return updates for each component based on the parsed parameters
    return (
        parameters.get("model/checkpoint", "stablediffusionapi/realistic-vision-v6.0-b1-inpaint"),
        parameters.get("scheduler", "DPMSolverMultistepScheduler"),
        parameters.get("controlnet", "None"),
        parameters.get("controlnet_strength", 1.0),
        parameters.get("seed", -1), 
        parameters.get("prompt", ""),             
        parameters.get("negative_prompt", ""),        
        int(parameters.get("width", 512)),              
        int(parameters.get("height", 768)),            
        int(parameters.get("steps", 25)),      
        float(parameters.get("cfg_scale", 7.0)),     
        int(parameters.get("clip_skip", 1)),               
        int(parameters.get("batch_size", 1)),
    )

    
    
    
def load_mask_to_inpaint(info):
    # Extract metadata
    parameters = extract_metadata(info)
    
    # Load the layer and background images
    mask_path = parameters.get("mask_path")
    background_path = parameters.get("output_path")
    
    try:
        mask_image = Image.open(mask_path)
        # Create layers using the removed background image
        layers = [remove_background(mask_image)]
        # if is_local:
        #     layers[0].save("layer.png")
        background = Image.open(background_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        layers = []  # Fallback to an empty list if there is an error
        background = None

    # Process the background image
    background = retrieve_mask(image=background)
    # if is_local:
    #     background.save("background.png")
    composite = create_composite(background, layers[0])
    # if is_local:
    #     composite.save("composite.png")
    
    # Return the images and layers dictionary
    return {
        "background": background,
        "layers": layers,
        "composite": composite
    }