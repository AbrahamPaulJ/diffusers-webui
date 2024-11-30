
from modules import IS_LOCAL 
from modules.image_processing import auto_select_dimensions, remove_background, create_composite, retrieve_mask

from PIL import Image 
import gradio as gr
import os
import random
import time

#Functions
def clear_gallery():
    return []

def make_visible():
    return gr.update(visible=True)

def i2i_make_visible(img):
    if img!=None:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def hide():
    return gr.update(visible=False)

def clear():
    return []

def send_to_inpaint(imgs):
    return imgs[0][0]
                        
def button_is_waiting():
    return gr.update(interactive=False, value="Loading...")

def generating():
    return gr.update(interactive=False, value="Generating...")
 
def toggle_mode(mode):
    if mode=="Image To Image":
        return gr.update(visible=False)
    if mode=="Inpaint":
        brush = gr.Brush(colors=["#000000"], color_mode='fixed',default_size=50)
        return gr.update(brush=brush, transforms=(), visible=True)
    if mode=="Outpaint":
        return gr.update(brush=False, transforms=('crop'),visible=True)
    
def toggle_mode_hide_i2i(mode,component_hide_count_i2i):
    if mode=="Image To Image":
        updates = [gr.update(visible=False) for _ in range(component_hide_count_i2i)] 
        return updates
    else:
        return [gr.update() for _ in range(component_hide_count_i2i)] 

def toggle_mode_show_i2i(mode,component_hide_count_i2i):
    if mode=="Image To Image":
        updates = [gr.update(visible=True) for _ in range(component_hide_count_i2i)] 
        return updates
    else:
        return [gr.update() for _ in range(component_hide_count_i2i)] 
    
def toggle_mode_hide_ip(mode,component_hide_count_ip):
    if mode=="Inpaint":
        updates = [gr.update(visible=False) for _ in range(component_hide_count_ip)] 
        return updates
    else:
        return [gr.update() for _ in range(component_hide_count_ip)] 

def toggle_mode_show_ip(mode,component_hide_count_ip):
    if mode=="Inpaint":
        updates = [gr.update(visible=True) for _ in range(component_hide_count_ip)] 
        return updates
    else:
        return [gr.update() for _ in range(component_hide_count_ip)] 
    
def toggle_mode_hide_op(mode,component_hide_count_op):
    if mode=="Outpaint":
        updates = [gr.update(visible=False) for _ in range(component_hide_count_op)] 
        return updates
    else:
        return [gr.update() for _ in range(component_hide_count_op)] 

def toggle_mode_show_op(mode,component_hide_count_op):
    if mode=="Outpaint":
        updates = [gr.update(visible=True) for _ in range(component_hide_count_op)] 
        return updates
    else:
        return [gr.update() for _ in range(component_hide_count_op)] 
    
def reset_seed():
    return -1

def random_seed():
    return random.randint(0, 2**32 - 1)

def recycle_seed(output_seed):
    return output_seed

def auto_dim(image):
    if image is not None:
        return auto_select_dimensions(image)
    else:
        return 512,512
    
def warn_no_image(img):
    if img==None:
        gr.Warning('Please upload an image.')
        
def change_controlnet(base_model):
    if base_model == "SDXL":
        return gr.update(
            choices=["None", "Canny - controlnet-canny-sdxl-1.0", "Depth - controlnet-depth-sdxl-1.0","OpenPose - controlnet-openpose-sdxl-1.0"], 
            value="None"
        )
    else:
        return gr.update(
            choices=["None", "Canny - control_v11p_sd15_canny", "Depth - control_v11f1p_sd15_depth", "OpenPose - control_v11p_sd15_openpose"], 
            value="None"
        )
    
def upload_control_img(controlnet_dropdown):
    if controlnet_dropdown!="None":
        return gr.update(visible=True),gr.update(visible=True)
    else:
        return gr.update(visible=False),gr.update(visible=False)
    
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
    
                      
import os

def get_metadata(image):
    if image is None:
        return "No metadata available.", gr.update(visible=False), gr.update(visible=False)
    
    # Extract metadata
    metadata = image.info  # Metadata dictionary for PNG images in PIL
    if not metadata:
        return "No metadata found in this image.", gr.update(visible=False), gr.update(visible=False)

    # print("Metadata extracted:", metadata)  # Debug print to check the metadata

    parameters_string = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    parameters = extract_metadata(parameters_string)
    # print("Extracted parameters:", parameters)  # Debug print to check the parameters
    
    output_path = parameters.get("output_path")
    if output_path:
        output_path = output_path.replace("\\", "/")  # Assign back to output_path
    # print("Output Path:", output_path)  # Check the output path
    
    t2i_btn_visible = True if (output_path and output_path.startswith("outputs/txt2img")) else False

    return "\n".join([f"{k}: {v}" for k, v in metadata.items()]), gr.update(visible=True), gr.update(visible=t2i_btn_visible)



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

def load_info_to_i2i(info,state):
    state^= 1
    parameters = extract_metadata(info)
    
    loras_used = parameters.get("loras_used", "")
    lora_list = loras_used.split(",") if loras_used else []
    lora_weights = parameters.get("lora_weights", "")
    background_path = parameters.get("output_path")
    if background_path:
        background = Image.open(background_path)
    else:
        background = None         
    
    # Return updates for each component based on the parsed parameters
    return (
        parameters.get("base_model", "SD"),
        parameters.get("model/checkpoint", "stablediffusionapi/realistic-vision-v6.0-b1-inpaint"),
        parameters.get("scheduler", "DPMSolverMultistepScheduler"),
        parameters.get("controlnet", "None"),
        parameters.get("controlnet_strength", 1.0),
        parameters.get("seed", -1), 
        parameters.get("prompt", ""),             
        parameters.get("negative_prompt", ""),        
        parameters.get("width", 512),              
        parameters.get("height", 768),            
        int(parameters.get("steps", 20)),  
        float(parameters.get("cfg_scale", 7.0)),     
        int(parameters.get("clip_skip", 1)),             
        int(parameters.get("batch_size", 1)),
        parameters.get("use_lora", "False") == "True",
        lora_list,
        lora_weights,
        background, 
        float(parameters.get("mask_blur", 0)), 
        parameters.get("fill_setting", "Inpaint Masked"),
        parameters.get("inpaint_mode", "Only Masked"), 
        parameters.get("maintain_aspect_ratio", "True") == "True", 
        parameters.get("post_process", "True") == "True",  
        float(parameters.get("denoise_strength", 0.75)),
        parameters.get("mode", "Image To Image"),
        parameters.get("image_positioned_at", "Center"),
        parameters.get("maximum_width/height", 768),
        state
    )
    

    
def load_mask_to_inpaint(info):
    time.sleep(0.2)
    # Extract metadata
    parameters = extract_metadata(info)
    if parameters.get("mode") == "Inpaint":
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
        # if IS_LOCAL:
        #     background.save("background.png")
        composite = create_composite(background, layers[0])
        # if IS_LOCAL:
        #     composite.save("composite.png")
        
        # Return the images and layers dictionary
        return {
            "background": background,
            "layers": layers,
            "composite": composite
        }
    else:
        return gr.update()
    
def load_info_to_t2i(info):
    # Extract metadata
    parameters = extract_metadata(info)
    loras_used = parameters.get("loras_used", "")
    lora_list = loras_used.split(",") if loras_used else []
    lora_weights = parameters.get("lora_weights", "")

    return (
        parameters.get("base_model", "SD"),
        parameters.get("model/checkpoint", "runwayml/stable-diffusion-v1-5"),
        parameters.get("scheduler", "DPM++_2M_KARRAS"),
        parameters.get("controlnet", "None"),
        float(parameters.get("controlnet_strength", 1.0)),
        parameters.get("seed", -1), 
        parameters.get("prompt", ""),             
        parameters.get("negative_prompt", ""),        
        parameters.get("width", 512),              
        parameters.get("height", 512),            
        int(parameters.get("steps", 20)),      
        float(parameters.get("cfg_scale", 7.0)),     
        int(parameters.get("clip_skip", 1)),               
        int(parameters.get("batch_size", 1)),
        parameters.get("hires_fix_2x", "False") == "True",
        parameters.get("use_lora", "False") == "True",
        lora_list,
        lora_weights  # Return lora_weights
    )
    
