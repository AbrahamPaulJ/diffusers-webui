# app.py

import gradio as gr
from modules.inpainting import *
from modules.txt2img import *
from modules.manage_models import model_dir
from modules.pipelines import PipelineManager
from torch import Generator
import random   
import re
from modules import IS_LOCAL

from modules.ui_functions import *

class StableDiffusionApp:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.pipeline_manager = PipelineManager(model_dir)
        # Load the default inpainting pipeline
        # self.pipeline_manager.load_pipeline()
        self.setup_gradio_interface()

    def setup_gradio_interface(self):
        """Create the Gradio interface with tabs for inpainting, text-to-image, and more."""
               
        with gr.Blocks(
            
            #            # button[aria-label="Clear"] {
            #     display: none !important;
            # }
            
            css=""" 
            .source-wrap { 
                display: none;  /* Hides the entire container with the buttons */ 
            }
            .source-selection { 
                display: none;  /* Hides the entire container with the buttons */ 
            } 
            .layer-wrap { 
                display: none;  /* Hides the layer button */ 
            }
            button[aria-label="Close"] {
                display: none !important;
            """,
            theme=gr.themes.Default(primary_hue="green", secondary_hue="pink")
        ) as iface:

            # Create the tabs
            with gr.Tabs():
                # Inpainting Tab
                with gr.TabItem("Image To Image"):
                    brush = gr.Brush(colors=["#000000"], color_mode='fixed',default_size=50)
                    load_status = gr.State(value=0)
                    
                    os.makedirs("models", exist_ok=True)
                    models_folder = "models"
                    model_choices = [
                        file.replace("models--", "", 1).replace("--", "/", 1) if file.startswith("models--") and "control" not in file.lower() 
                        else file
                        for file in os.listdir(models_folder)
                        if (file.endswith((".safetensors", ".ckpt")) or 
                            (os.path.isdir(os.path.join(models_folder, file)) and 
                            "control" not in file.lower() and not file.startswith(".")))
                    ]

                    
                    with gr.Row(equal_height=True):
                        # Dropdown for selecting inpainting checkpoint
                        checkpoint_dropdown = gr.Dropdown(
                            label="Select Checkpoint", 
                            choices = model_choices, 
                            value="stablediffusionapi/realistic-vision-v6.0-b1-inpaint"
                        )
                            
                        # Dropdown for selecting scheduler
                        scheduler_dropdown = gr.Dropdown(
                            label="Select Scheduler", 
                            choices=[
                                "DPM++_2M_KARRAS", "EULER_A", "EULER", "DDIM", "DDPM", "DEIS", 
                                "DPM2", "DPM2-A", "DPM++_2S", "DPM++_2M", "DPM++_SDE", 
                                "DPM++_SDE_KARRAS", "UNIPC", "HEUN", "HEUN_KARRAS", 
                                "LMS", "LMS_KARRAS", "PNDM"
                            ], 
                            value="DPM++_2M_KARRAS",
                        )

                        
                        os.makedirs("loras", exist_ok=True)
                        loras_folder = "loras"
                        lora_choices = [file for file in os.listdir(loras_folder) if file.endswith(".safetensors")]
                        lora_default_value = lora_choices[0] if lora_choices else None
                        
                        with gr.Column():
                            use_lora = gr.Checkbox(label="Use LoRAs", value=False)
                            
                            lora_dropdown = gr.Dropdown(
                                label="Select LoRAs",
                                choices=lora_choices if lora_choices else [],
                                value=lora_default_value,
                                visible=False,
                                multiselect=True
                            )                     


                        #lora_refresh_btn = gr.Button("🔄")
                                       
                    with gr.Row():
                        inpaint_input_image = gr.Image(type="pil", label="Input Image", height=600)
                        inpaint_mask = gr.ImageEditor(type="pil", label="Mask Editor", height=600, brush=brush, transforms=(), sources=('clipboard'), placeholder="Mask Preview", layers=False)
                        output_image = gr.Gallery(type="pil", label="Generated Image(s)", height=600, selected_index=0, columns=1, rows=1, visible=False)

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            with gr.Row():
                                width = gr.Dropdown(label="Width", choices=[512, 768, 1024], value=512, allow_custom_value=True)
                                height = gr.Dropdown(label="Height", choices=[512, 768, 1024], value=512, allow_custom_value=True)
                            mode = gr.Radio(["Inpaint", "Outpaint"], value="Inpaint", label = "Mode")
                            mask_blur = gr.Slider(label="Mask Blur", minimum=0, maximum=40, value=0, step=0.1)

                        with gr.Column(scale=1):
                            fill_setting = gr.Radio(label="Mask", choices=["Inpaint Masked", "Inpaint Not Masked"], value="Inpaint Masked")
                            steps = gr.Slider(label="Number of Steps", value=25, maximum=50, minimum=1,step=1)
                            denoise_strength = gr.Slider(label="Denoise Strength", minimum=0, maximum=1, value=1, step=0.01)
                            outpaint_img_pos = gr.Radio(label="Image Positioned at:", choices=["Center", "Top", "Bottom"], value="Center", visible=False)
                            outpaint_max_dim = gr.Dropdown(label="Maximum Width/Height", choices=[512, 768, 1024], value=768, visible=False, allow_custom_value=True)
                            
                        with gr.Column(scale=1):
                            maintain_aspect_ratio = gr.Checkbox(label="Maintain Aspect Ratio (Auto Padding)", value=True)
                            post_process = gr.Checkbox(label="Post-Processing", value=True)
                            controlnet_dropdown = gr.Dropdown(
                            label="Select Controlnet", 
                            choices=["None", "Canny - lllyasviel/control_v11p_sd15_canny", "Depth - lllyasviel/control_v11f1p_sd15_depth","OpenPose - lllyasviel/control_v11p_sd15_openpose"], 
                            value="None"
                        )
                            controlnet_strength = gr.Slider(label="ControlNet Strength", minimum=0, maximum=1, value=1, step=0.01)
                            cfg_scale =  gr.Slider(label="CFG Scale", value=7, minimum=0, maximum=20, step=0.5)

                        with gr.Column(scale=1):
                            generate_button = gr.Button(value="Generate Image")
                            clear_button = gr.Button("Clear Image")
                            batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, value=1, step=1)
                            #upscalefour = gr.Button("Upscale to 4x")
                            output_seed = gr.Textbox(value=-1, label="Output Seed", interactive=False, show_copy_button=True,show_label=True, visible=False)

                    with gr.Row(equal_height=True):
                        with gr.Column():
                            lora_prompt = gr.Textbox(label="Adjust LoRA Weights", lines=1, visible=False)
                            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=6)
                        negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here...", lines=6)

                    with gr.Row(equal_height=True):
                        with gr.Group():
                            seed = gr.Number(label="Seed", value=-1)
                            with gr.Column():
                                reset_seed_btn = gr.Button("↺", size='lg')
                                random_seed_btn = gr.Button("🎲", size='lg')
                                reuse_seed_btn = gr.Button("♻️", size='lg')
                                
                        clip_skip = gr.Dropdown(label="CLIP Skip", choices=[0, 1, 2], value=1)

   
                    toggle_mode_hide_components = [fill_setting, maintain_aspect_ratio]
                    component_hide_len = len(toggle_mode_hide_components)
                    component_hide_count = gr.State(value=component_hide_len)
                    
                    toggle_mode_show_components = [outpaint_img_pos,outpaint_max_dim]
                    component_show_len = len(toggle_mode_show_components)
                    component_show_count = gr.State(value=component_show_len)
                    
                    # Listen for events
                    inpaint_input_image.change(fn=retrieve_mask, inputs=[mode, outpaint_img_pos, inpaint_input_image], outputs=[inpaint_mask])
                    inpaint_input_image.change(fn=auto_dim, inputs=[inpaint_input_image],outputs=[width,height])
                    outpaint_img_pos.change(fn=retrieve_mask, inputs=[mode, outpaint_img_pos, inpaint_input_image], outputs=[inpaint_mask])
                    mode.change(fn=retrieve_mask, inputs=[mode, outpaint_img_pos, inpaint_input_image], outputs=[inpaint_mask])
                    mode.change(fn=toggle_mode,inputs=[mode], outputs=inpaint_mask)
                    mode.change(fn=toggle_mode_hide,inputs=[mode,component_hide_count], outputs=toggle_mode_hide_components)
                    mode.change(fn=toggle_mode_show,inputs=[mode,component_show_count], outputs=toggle_mode_show_components)
                           
                    gr.on(
                        triggers=[checkpoint_dropdown.change, controlnet_dropdown.change, scheduler_dropdown.change],
                        fn=button_is_waiting,
                        inputs=None,
                        outputs=generate_button
                    )
                            
                    gr.on(
                        triggers=[checkpoint_dropdown.change, controlnet_dropdown.change, scheduler_dropdown.change],
                        fn=self.load_inpaint,
                        inputs=[checkpoint_dropdown, scheduler_dropdown, controlnet_dropdown, use_lora, lora_prompt],
                        outputs=generate_button
                    )

                    reset_seed_btn.click(fn=reset_seed, outputs=seed)
                    random_seed_btn.click(fn=random_seed, outputs=seed)
                    reuse_seed_btn.click(fn=recycle_seed,inputs=output_seed, outputs=seed)
                    
                    generate_button.click(
                        fn=warn_no_image,
                        inputs=inpaint_input_image,
                        outputs=None 
                    )
                    
                    generate_button.click(
                        fn=generating,
                        inputs=None,
                        outputs=generate_button  
                    )

                    generate_button.click(
                        fn=clear_gallery,
                        inputs=None,
                        outputs=output_image
                    )

                    generate_button.click(
                        fn=i2i_make_visible,
                        inputs=inpaint_input_image,
                        outputs=[output_seed]
                    )

                    generate_button.click(
                        fn=i2i_make_visible,
                        inputs=inpaint_input_image,
                        outputs=[output_image]
                    )
                    
                    generate_button.click(
                        fn=self.load_and_seed_gen_inpaint,
                        inputs=[load_status, checkpoint_dropdown, scheduler_dropdown, controlnet_dropdown,use_lora, lora_prompt],
                        outputs=[load_status]
                    )
                                                                   
                    load_status.change(
                        fn=self.seed_and_gen_inpaint_image,
                        inputs=[controlnet_dropdown, seed,inpaint_input_image, prompt, negative_prompt, width, height, steps, cfg_scale, clip_skip, inpaint_mask, fill_setting, maintain_aspect_ratio, post_process, denoise_strength, batch_size, mask_blur, mode, outpaint_img_pos, outpaint_max_dim, controlnet_strength, use_lora, lora_dropdown, lora_prompt],
                        outputs=[generate_button, output_seed, output_image]
                    )
                    
                    use_lora.change(fn=using_lora, inputs=use_lora, outputs=[lora_dropdown,lora_prompt])
                    use_lora.change(fn=lora_to_prompt_cb,inputs=[steps,lora_dropdown],outputs=lora_prompt)
                    lora_dropdown.change(fn=lora_to_prompt, inputs=lora_dropdown,outputs=lora_prompt)
                    
                    clear_button.click(fn=hide, inputs=None, outputs=output_image)
                    
                    #lora_refresh_btn.click(fn=update_lora_dropdown, outputs=lora_dropdown)
                    #upscalefour.click(fn=upscale, inputs=output_image, outputs=output_image)

                # Text-to-Image Tab
                with gr.TabItem("Text-to-Image"):
                    txt2img_load_status = gr.State(value=0)
                    with gr.Row(equal_height=True):

                        # Dropdown for selecting text-to-image model
                        txt2img_checkpoint_dropdown = gr.Dropdown(
                            label="Select Text-to-Image Model", 
                            choices = model_choices, 
                            value="runwayml/stable-diffusion-v1-5",
                        )
                        # Dropdown for selecting scheduler
                        txt2img_scheduler_dropdown = gr.Dropdown(
                            label="Select Scheduler", 
                            choices=[
                                "DPM++_2M_KARRAS", "EULER_A", "EULER", "DDIM", "DDPM", "DEIS", 
                                "DPM2", "DPM2-A", "DPM++_2S", "DPM++_2M", "DPM++_SDE", 
                                "DPM++_SDE_KARRAS", "UNIPC", "HEUN", "HEUN_KARRAS", 
                                "LMS", "LMS_KARRAS", "PNDM"
                            ], 
                            value="DPM++_2M_KARRAS",
                        )
                        with gr.Column():
                            txt2img_use_lora = gr.Checkbox(label="Use LoRAs", value=False)
                            
                            txt2img_lora_dropdown = gr.Dropdown(
                                label="Select LoRAs",
                                choices=lora_choices if lora_choices else [],
                                value=lora_default_value,
                                visible=False,
                                multiselect=True
                            )
                        
                    with gr.Row(equal_height=True): 
                        with gr.Column():
                                with gr.Row():
                                    txt2img_width = gr.Dropdown(label="Width", choices=[512, 768, 1024], value=512,allow_custom_value=True)
                                    txt2img_height = gr.Dropdown(label="Height", choices=[512, 768, 1024], value=512,allow_custom_value=True)
                                with gr.Row():
                                    txt2img_steps = gr.Slider(label="Number of Steps", value=25, maximum=50, minimum=1,step=1)
                                    txt2img_cfg_scale = gr.Slider(label="CFG Scale", value=7, minimum=0, maximum=20, step=0.5)
                                    txt2img_hires_fix = gr.Checkbox(label="Hires. fix - Latent 2x", value=False)
                                with gr.Row(equal_height=True):
                                    with gr.Group():
                                            txt2img_seed = gr.Number(label="Seed", value=-1)                                   
                                            txt2img_reset_seed_btn = gr.Button("↺", size='lg')
                                            txt2img_random_seed_btn = gr.Button("🎲", size='lg')
                                            txt2img_reuse_seed_btn = gr.Button("♻️", size='lg')
                                    with gr.Column():        
                                        txt2img_clip_skip = gr.Dropdown(label="CLIP Skip", choices=[0, 1, 2], value=1)
                                        txt2img_batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, value=1, step=1)
                                
                                with gr.Row():                          
                                    txt2img_controlnet_dropdown = gr.Dropdown(
                                            label="Select Controlnet", 
                                            choices=["None", "Canny - lllyasviel/control_v11p_sd15_canny", "Depth - lllyasviel/control_v11f1p_sd15_depth","OpenPose - lllyasviel/control_v11p_sd15_openpose"], 
                                            value="None"
                                        )
                                    txt2img_controlnet_strength = gr.Slider(label="ControlNet Strength", minimum=0, maximum=1, value=1, step=0.01, visible=False)
                                txt2img_control_image = gr.Image(type="pil", label="Control Image", height=300, visible=False)
                                
                                    
                        txt2img_output_image = gr.Gallery(type="pil", label="Generated Image(s)", selected_index=0, columns=1, rows=1, visible=True)
                                       
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            txt2img_lora_prompt = gr.Textbox(label="Adjust LoRA Weights", lines=1, visible=False)
                            txt2img_prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=8)
                        with gr.Column():
                            with gr.Row():
                                txt2img_generate_button = gr.Button("Generate Image")
                                txt2img_clear_button = gr.Button("Clear Image")
                            with gr.Row(): 
                                txt2img_output_seed = gr.Textbox(value=-1, label="Output Seed", interactive=False, show_copy_button=True,show_label=True, visible=False)   
                            txt2img_negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here...", lines=8)
                            

                    # Listeners
                    gr.on(
                        triggers=[txt2img_checkpoint_dropdown.change, txt2img_controlnet_dropdown.change, txt2img_scheduler_dropdown.change],
                        fn=button_is_waiting,
                        inputs=None,
                        outputs=txt2img_generate_button
                    )
                            
                    gr.on(
                        triggers=[txt2img_checkpoint_dropdown.change, txt2img_controlnet_dropdown.change, txt2img_scheduler_dropdown.change],
                        fn=self.load_txt2img,
                        inputs=[txt2img_checkpoint_dropdown, txt2img_scheduler_dropdown, txt2img_controlnet_dropdown, txt2img_use_lora, txt2img_lora_prompt],
                        outputs=txt2img_generate_button
                    )

                    txt2img_reset_seed_btn.click(fn=reset_seed, outputs=txt2img_seed)
                    txt2img_random_seed_btn.click(fn=random_seed, outputs=txt2img_seed)
                    txt2img_reuse_seed_btn.click(fn=recycle_seed,inputs=txt2img_output_seed, outputs=txt2img_seed)
                    
                    txt2img_generate_button.click(
                        fn=clear_gallery,
                        inputs=None,
                        outputs=txt2img_output_image
                    )
                    
                    txt2img_generate_button.click(
                        fn=generating,
                        inputs=None,
                        outputs=txt2img_generate_button  
                    )
                    
                    txt2img_generate_button.click(
                        fn=make_visible,
                        inputs=None,
                        outputs=[txt2img_output_seed]
                    )
                    
                    txt2img_generate_button.click(
                        fn=make_visible,
                        inputs=None,
                        outputs=[txt2img_output_image]
                    )
                    
                    txt2img_generate_button.click(
                        fn=self.load_and_seed_gen_txt2img,
                        inputs=[txt2img_load_status, txt2img_checkpoint_dropdown, txt2img_scheduler_dropdown, txt2img_controlnet_dropdown, txt2img_use_lora, txt2img_lora_prompt],
                        outputs=[txt2img_load_status]
                    )
                                                                   
                    txt2img_load_status.change(
                        fn=self.seed_and_gen_tx2img_image,
                         inputs=[txt2img_controlnet_dropdown, txt2img_seed, txt2img_prompt, txt2img_negative_prompt, txt2img_width, txt2img_height, txt2img_steps, txt2img_cfg_scale, txt2img_clip_skip,txt2img_control_image, txt2img_batch_size, txt2img_controlnet_strength, txt2img_hires_fix, txt2img_use_lora, txt2img_lora_dropdown, txt2img_lora_prompt],
                         outputs=[txt2img_generate_button, txt2img_output_seed, txt2img_output_image]
                    )
                    
                    txt2img_controlnet_dropdown.change(fn=upload_control_img,inputs=txt2img_controlnet_dropdown,outputs=[txt2img_control_image,txt2img_controlnet_strength])
                    txt2img_use_lora.change(fn=using_lora, inputs=txt2img_use_lora, outputs=[txt2img_lora_dropdown,txt2img_lora_prompt])
                    txt2img_use_lora.change(fn=lora_to_prompt_cb,inputs=[txt2img_steps,txt2img_lora_dropdown],outputs=txt2img_lora_prompt)
                    txt2img_lora_dropdown.change(fn=lora_to_prompt, inputs=txt2img_lora_dropdown,outputs=txt2img_lora_prompt)
                    txt2img_clear_button.click(fn=clear, inputs=None, outputs=txt2img_output_image)          
                
                with gr.TabItem("PNG Info"):
                    with gr.Row():
                        png_input_image = gr.Image(type="pil", label="Input Image", height=600)
                        with gr.Column():
                            png_info = gr.Textbox(label="Generation Parameters",lines=25, show_copy_button=True,show_label=True)
                            with gr.Row(equal_height=True):
                                info_to_inpaint_btn = gr.Button("Send Parameters to Inpaint Tab",visible=False)   
                                info_to_txt2img_btn = gr.Button("Send Parameters to Text to Image Tab",visible=False)  
                                state= gr.State(value=0)                            
                                               
                    #Listeners
                    png_input_image.change(fn=get_metadata, inputs=png_input_image, outputs=[png_info, info_to_inpaint_btn, info_to_txt2img_btn])
                    
                    info_to_inpaint_btn.click(
                    fn=load_info_to_inpaint,
                    inputs=[png_info,state],
                    outputs=[
                        checkpoint_dropdown,
                        scheduler_dropdown,
                        controlnet_dropdown,
                        controlnet_strength,
                        seed,
                        inpaint_input_image,
                        prompt,  
                        negative_prompt,  
                        width,  
                        height,  
                        steps, 
                        mask_blur,
                        cfg_scale,  
                        clip_skip, 
                        fill_setting, 
                        maintain_aspect_ratio,  
                        post_process,   
                        denoise_strength, 
                        batch_size,
                        mode,
                        outpaint_img_pos,
                        outpaint_max_dim,
                        use_lora,
                        lora_dropdown,
                        lora_prompt,
                        state
                    ])
                    
                    info_to_txt2img_btn.click(
                    fn=load_info_to_txt2img,
                    inputs=png_info,
                    outputs=[
                        txt2img_checkpoint_dropdown,
                        txt2img_scheduler_dropdown,
                        txt2img_controlnet_dropdown,
                        txt2img_controlnet_strength,
                        txt2img_seed,
                        txt2img_prompt,  
                        txt2img_negative_prompt,  
                        txt2img_width,  
                        txt2img_height,  
                        txt2img_steps, 
                        txt2img_cfg_scale,  
                        txt2img_clip_skip, 
                        txt2img_batch_size,
                        txt2img_hires_fix,
                        txt2img_use_lora,
                        txt2img_lora_dropdown,
                        txt2img_lora_prompt
                    ])
                    
                    
                    state.change(
                        fn=load_mask_to_inpaint,
                        inputs=png_info,
                        outputs=[inpaint_mask]
                    )                                    

        iface.queue().launch(share= not IS_LOCAL)
        
        
    def load_txt2img(self, checkpoint, scheduler, controlnet, use_lora, lora_dict, generate_button_clicked=False):
        """Load the pipeline based on changes in the checkpoint or ControlNet selection."""
        if generate_button_clicked==False:
            lora_dict = self.parse_lora_prompt(lora_dict)
        # Load pipeline with the determined parameters
        self.pipeline_manager.load_pipeline(checkpoint, "txt2img", scheduler, controlnet_type=controlnet, use_lora=use_lora, lora_dict=lora_dict)
        return gr.update(interactive=True, value="Generate Image") if generate_button_clicked is False else gr.update(interactive=False, value="Generating...")
    
    def load_and_seed_gen_txt2img(self, load_status, checkpoint, scheduler, controlnet, use_lora, lora_prompt):
        """Load models and generate when generate button clicked."""
        
        lora_dict=self.parse_lora_prompt(lora_prompt)
        load_status ^= 1
        self.load_txt2img(checkpoint, scheduler, controlnet,use_lora, lora_dict, generate_button_clicked=True)
        
        return load_status
    
    
    def seed_and_gen_tx2img_image( self, controlnet_name, seed, *args):
        """Generate an inpainted image after loading the appropriate model."""
        
        if seed is None or seed == "" or seed == -1:
            # Generate a random seed if not provided
            seed = random.randint(0, 2**32 - 1)  # Generate a random seed
            generator = Generator().manual_seed(seed)  # Create a generator with the random seed
        else:
            seed = int(seed)
        generator = Generator().manual_seed(seed)  # Create a generator with the specified seed
        # Generate the inpainted image with the loaded pipeline
        
        return gr.update(interactive=True, value="Generate Image"), str(seed), generate_txt2img_image(
            self.pipeline_manager, controlnet_name, seed, generator,
            *args) 
            
    def load_inpaint(self, checkpoint, scheduler, controlnet, use_lora, lora_dict, generate_button_clicked=False):
        """Load the pipeline based on changes in the checkpoint or ControlNet selection."""
        if generate_button_clicked==False:
            lora_dict = self.parse_lora_prompt(lora_dict)
        # Load pipeline with the determined parameters
        self.pipeline_manager.load_pipeline(checkpoint, "inpainting", scheduler, controlnet_type=controlnet, use_lora=use_lora, lora_dict=lora_dict)
        return gr.update(interactive=True, value="Generate Image") if generate_button_clicked is False else gr.update(interactive=False, value="Generating...")
    
    def load_and_seed_gen_inpaint(self, load_status, checkpoint, scheduler, controlnet, use_lora, lora_prompt):
        """Load models and generate when generate button clicked."""
        
        lora_dict=self.parse_lora_prompt(lora_prompt)
        load_status ^= 1
        self.load_inpaint(checkpoint, scheduler, controlnet, use_lora, lora_dict, generate_button_clicked=True)
        
        return load_status
        
    
    def seed_and_gen_inpaint_image(self, controlnet_name, seed, input_image, *args):
        """Generate an inpainted image after loading the appropriate model."""
        
        if input_image is None:
            return gr.update(interactive=True, value="Generate Image"),str(seed), []

        if seed is None or seed == "" or seed == -1:
            # Generate a random seed if not provided
            seed = random.randint(0, 2**32 - 1)  # Generate a random seed
            generator = Generator().manual_seed(seed)  # Create a generator with the random seed
        else:
            seed = int(seed)
        generator = Generator().manual_seed(seed)  # Create a generator with the specified seed
        # Generate the inpainted image with the loaded pipeline
        
        return gr.update(interactive=True, value="Generate Image"), str(seed), generate_inpaint_image(
            self.pipeline_manager, controlnet_name, seed, input_image, generator,
            *args)
        

    def parse_lora_prompt(self, lora_prompt):
        # Return None if the lora_prompt is an empty string
        if lora_prompt == "":
            return None
        
        # Find all matches for the LoRA name and scale using regular expressions
        lora_matches = re.findall(r"<lora:([^:]+):([\d.]+)>", lora_prompt)
        
        # Convert matches to dictionary with filename as key and scale as value
        lora_dict = {f"loras/{name}.safetensors": float(scale) for name, scale in lora_matches}

        return lora_dict

# Main execution
if __name__ == "__main__":
    
    app = StableDiffusionApp(model_dir)

