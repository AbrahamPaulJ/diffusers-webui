# app.py

import gradio as gr
from modules import IS_LOCAL, DEVICE
from modules.pipelines import PipelineManager
from modules.img2img import *
from modules.txt2img import *
from modules.ui_functions import *

from torch import Generator
import random   
import re

class StableDiffusionApp:
    def __init__(self):
        self.pipeline_manager = PipelineManager()
        # Load the default pipeline
        # self.pipeline_manager.load_pipeline()
        self.setup_gradio_interface()

    def setup_gradio_interface(self):
        """Create the Gradio interface with tabs for image-to-image, text-to-image, and more."""
               
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
                # Image To Image Tab
                with gr.TabItem("Image To Image"):
                    brush = gr.Brush(colors=["#000000"], color_mode='fixed',default_size=50)
                    load_status = gr.State(value=0)
 
                    os.makedirs("models", exist_ok=True)
                    models_folder = "models"
                    checkpoint_choices = [
                        file.replace("models--", "", 1).replace("--", "/", 1) if file.startswith("models--") and "control" not in file.lower() 
                        else file
                        for file in os.listdir(models_folder)
                        if (file.endswith((".safetensors", ".ckpt")) or 
                            (os.path.isdir(os.path.join(models_folder, file)) and 
                            "control" not in file.lower() and not file.startswith(".")))
                    ]
                    default_checkpoints = ["stablediffusionapi/realistic-vision-v6.0-b1-inpaint", "runwayml/stable-diffusion-v1-5"]

                    for checkpoint in default_checkpoints:
                        if checkpoint not in checkpoint_choices:
                            checkpoint_choices.append(checkpoint)

                    with gr.Row(equal_height=True):
                                            
                        base_model_dropdown = gr.Radio(
                                label="Base Model",
                                choices=["SD","SDXL"],
                                value="SD", scale=0, min_width=200
                            )

                        checkpoint_dropdown = gr.Dropdown(
                            label="Select Checkpoint", 
                            choices = checkpoint_choices, 
                            value="stablediffusionapi/realistic-vision-v6.0-b1-inpaint"
                        )

                        scheduler_dropdown = gr.Dropdown(
                            label="Select Scheduler", 
                            choices=[
                                "DPM++_2M_KARRAS","DPM++_SDE", 
                                "DPM++_SDE_KARRAS", "EULER_A", "EULER", "DDIM", "DDPM", "DEIS", 
                                "DPM2", "DPM2-A", "DPM++_2S", "DPM++_2M", "UNIPC", "HEUN", "HEUN_KARRAS", 
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
                        input_image = gr.Image(type="pil", label="Input Image", height=600)
                        inpaint_mask = gr.ImageEditor(type="pil", label="Mask Editor", height=600, brush=brush, transforms=(), sources=('clipboard'), placeholder="Mask Preview", layers=False)
                        output_image = gr.Gallery(type="pil", label="Generated Image(s)", height=600, selected_index=0, columns=1, rows=1, visible=False)

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            with gr.Row():
                                width = gr.Dropdown(label="Width", choices=[512, 768, 1024], value=512, allow_custom_value=True)
                                height = gr.Dropdown(label="Height", choices=[512, 768, 1024], value=512, allow_custom_value=True)
                            mode = gr.Radio(["Image To Image", "Inpaint", "Outpaint"], value="Inpaint", label = "Mode")
                            mask_blur = gr.Slider(label="Mask Blur", minimum=0, maximum=40, value=0, step=0.1)

                        with gr.Column(scale=1):
                            fill_setting = gr.Radio(label="Mask", choices=["Inpaint Masked", "Inpaint Not Masked"], value="Inpaint Masked")
                            mask_crop = gr.Radio(label="Inpaint Mode", choices=["Whole Picture", "Only Masked"], value="Only Masked")
                            steps = gr.Slider(label="Number of Steps", value=20, maximum=50, minimum=1,step=1)
                            denoise_strength = gr.Slider(label="Denoise Strength", minimum=0, maximum=1, value=0.75, step=0.01)
                            outpaint_img_pos = gr.Radio(label="Image Positioned at:", choices=["Center", "Top", "Bottom"], value="Center", visible=False)
                            outpaint_max_dim = gr.Dropdown(label="Maximum Width/Height", choices=[512, 768, 1024], value=768, visible=False, allow_custom_value=True)
                            
                        with gr.Column(scale=1):
                            maintain_aspect_ratio = gr.Checkbox(label="Maintain Aspect Ratio (Auto Padding)", value=True)
                            post_process = gr.Checkbox(label="Post-Processing", value=True)
                            controlnet_dropdown = gr.Dropdown(
                            label="Select Controlnet", 
                            choices=["None", "Canny - control_v11p_sd15_canny", "Depth - control_v11f1p_sd15_depth","OpenPose - control_v11p_sd15_openpose"], 
                            value="None"
                        )
                            controlnet_strength = gr.Slider(label="ControlNet Strength", minimum=0, maximum=1, value=1, step=0.01)
                            cfg_scale =  gr.Slider(label="CFG Scale", value=7, minimum=0, maximum=20, step=0.5)

                        with gr.Column(scale=1):
                            generate_button = gr.Button(value="Generate Image")
                            clear_button = gr.Button("Clear Output")
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
                        
                    toggle_mode_show_components_i2i = [maintain_aspect_ratio]
                    component_show_count_i2i = gr.State(value=len(toggle_mode_show_components_i2i))

                    toggle_mode_hide_components_i2i = [inpaint_mask,mask_blur,fill_setting,post_process, mask_crop]
                    component_hide_count_i2i = gr.State(value=len(toggle_mode_hide_components_i2i))
                    
                    toggle_mode_show_components_op = [mask_blur,post_process,outpaint_img_pos,outpaint_max_dim]
                    component_show_count_op = gr.State(value=len(toggle_mode_show_components_op))
                    
                    toggle_mode_hide_components_op = [fill_setting, maintain_aspect_ratio, mask_crop]
                    component_hide_count_op = gr.State(value=len(toggle_mode_hide_components_op))
                    
                    toggle_mode_show_components_ip = [mask_blur,post_process,fill_setting,maintain_aspect_ratio, mask_crop]
                    component_show_count_ip = gr.State(value=len(toggle_mode_show_components_ip))
                    
                    toggle_mode_hide_components_ip = [outpaint_img_pos,outpaint_max_dim]
                    component_hide_count_ip = gr.State(value=len(toggle_mode_hide_components_ip))
                    
                    # Listen for events
                    input_image.change(fn=retrieve_mask, inputs=[mode, outpaint_img_pos, input_image], outputs=[inpaint_mask])
                    input_image.change(fn=auto_dim, inputs=[input_image],outputs=[width,height])
                    outpaint_img_pos.change(fn=retrieve_mask, inputs=[mode, outpaint_img_pos, input_image], outputs=[inpaint_mask])
                    mode.change(fn=retrieve_mask, inputs=[mode, outpaint_img_pos, input_image], outputs=[inpaint_mask])
                    mode.change(fn=toggle_mode,inputs=[mode], outputs=inpaint_mask)
                    mode.change(fn=toggle_mode_hide_i2i, inputs=[mode,component_hide_count_i2i], outputs = toggle_mode_hide_components_i2i)
                    mode.change(fn=toggle_mode_show_i2i, inputs=[mode,component_show_count_i2i], outputs = toggle_mode_show_components_i2i) 
                    mode.change(fn=toggle_mode_hide_ip, inputs=[mode,component_hide_count_ip], outputs = toggle_mode_hide_components_ip)
                    mode.change(fn=toggle_mode_show_ip, inputs=[mode,component_show_count_ip], outputs = toggle_mode_show_components_ip)
                    mode.change(fn=toggle_mode_hide_op, inputs=[mode,component_hide_count_op], outputs = toggle_mode_hide_components_op)
                    mode.change(fn=toggle_mode_show_op, inputs=[mode,component_show_count_op], outputs = toggle_mode_show_components_op)
                           
                    gr.on(
                        triggers=[base_model_dropdown.change, checkpoint_dropdown.change, controlnet_dropdown.change, scheduler_dropdown.change, mode.change],
                        fn=button_is_waiting,
                        inputs=None,
                        outputs=generate_button
                    )
                            
                    gr.on(
                        triggers=[base_model_dropdown.change, checkpoint_dropdown.change, controlnet_dropdown.change, scheduler_dropdown.change, mode.change],
                        fn=self.load_i2i,
                        inputs=[base_model_dropdown, checkpoint_dropdown, scheduler_dropdown, controlnet_dropdown, use_lora, lora_prompt, mode],
                        outputs=generate_button
                    )

                    reset_seed_btn.click(fn=reset_seed, outputs=seed)
                    random_seed_btn.click(fn=random_seed, outputs=seed)
                    reuse_seed_btn.click(fn=recycle_seed,inputs=output_seed, outputs=seed)
                    
                    generate_button.click(
                        fn=warn_no_image,
                        inputs=input_image,
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
                        inputs=input_image,
                        outputs=[output_seed]
                    )

                    generate_button.click(
                        fn=i2i_make_visible,
                        inputs=input_image,
                        outputs=[output_image]
                    )
                    
                    generate_button.click(
                        fn=self.load_and_seed_gen_i2i,
                        inputs=[load_status, base_model_dropdown, checkpoint_dropdown, scheduler_dropdown, controlnet_dropdown,use_lora, lora_prompt, mode],
                        outputs=[load_status]
                    )
                                                                   
                    load_status.change(
                        fn=self.seed_and_gen_i2i,
                        inputs=[base_model_dropdown, controlnet_dropdown, seed,input_image, prompt, negative_prompt, width, height, steps, cfg_scale, clip_skip, inpaint_mask, fill_setting, maintain_aspect_ratio, post_process, denoise_strength, batch_size, mask_blur, mode, outpaint_img_pos, outpaint_max_dim, controlnet_strength, use_lora, lora_dropdown, lora_prompt, mask_crop],
                        outputs=[generate_button, output_seed, output_image]
                    )
                    
                    base_model_dropdown.change(fn=change_controlnet, inputs=base_model_dropdown,outputs=controlnet_dropdown)
                    use_lora.change(fn=using_lora, inputs=use_lora, outputs=[lora_dropdown,lora_prompt])
                    use_lora.change(fn=lora_to_prompt_cb,inputs=[steps,lora_dropdown],outputs=lora_prompt)
                    lora_dropdown.change(fn=lora_to_prompt, inputs=lora_dropdown,outputs=lora_prompt)
                    
                    clear_button.click(fn=hide, inputs=None, outputs=output_image)
                    
                    #lora_refresh_btn.click(fn=update_lora_dropdown, outputs=lora_dropdown)
                    #upscalefour.click(fn=upscale, inputs=output_image, outputs=output_image)

                # Text To Image Tab
                with gr.TabItem("Text To Image"):
                    
                    t2i_load_status = gr.State(value=0)
                    
                    with gr.Row(equal_height=True):
                        t2i_base_model_dropdown = gr.Radio(
                            label="Base Model",
                            choices=["SD","SDXL"],
                            value="SD", scale=0, min_width=200
                        )

                        # Dropdown for selecting text-to-image model
                        t2i_checkpoint_dropdown = gr.Dropdown(
                            label="Select Checkpoint", 
                            choices = checkpoint_choices, 
                            value="runwayml/stable-diffusion-v1-5",
                        )
                        # Dropdown for selecting scheduler
                        t2i_scheduler_dropdown = gr.Dropdown(
                            label="Select Scheduler", 
                            choices=[
                                "DPM++_2M_KARRAS","DPM++_SDE", 
                                "DPM++_SDE_KARRAS", "EULER_A", "EULER", "DDIM", "DDPM", "DEIS", 
                                "DPM2", "DPM2-A", "DPM++_2S", "DPM++_2M", "UNIPC", "HEUN", "HEUN_KARRAS", 
                                "LMS", "LMS_KARRAS", "PNDM"
                            ], 
                            value="DPM++_2M_KARRAS",
                        )
                        with gr.Column():
                            t2i_use_lora = gr.Checkbox(label="Use LoRAs", value=False)
                            
                            t2i_lora_dropdown = gr.Dropdown(
                                label="Select LoRAs",
                                choices=lora_choices if lora_choices else [],
                                value=lora_default_value,
                                visible=False,
                                multiselect=True
                            )
                        
                    with gr.Row(equal_height=True): 
                        with gr.Column():
                                with gr.Row():
                                    t2i_width = gr.Dropdown(label="Width", choices=[512, 768, 1024], value=512,allow_custom_value=True)
                                    t2i_height = gr.Dropdown(label="Height", choices=[512, 768, 1024], value=512,allow_custom_value=True)
                                with gr.Row():
                                    t2i_steps = gr.Slider(label="Number of Steps", value=20, maximum=50, minimum=1,step=1)
                                    t2i_cfg_scale = gr.Slider(label="CFG Scale", value=7, minimum=0, maximum=20, step=0.5)
                                    t2i_hires_fix = gr.Checkbox(label="Hires. fix - Latent 2x", value=False)
                                with gr.Row(equal_height=True):
                                    with gr.Group():
                                            t2i_seed = gr.Number(label="Seed", value=-1)                                   
                                            t2i_reset_seed_btn = gr.Button("↺", size='lg')
                                            t2i_random_seed_btn = gr.Button("🎲", size='lg')
                                            t2i_reuse_seed_btn = gr.Button("♻️", size='lg')
                                    with gr.Column():        
                                        t2i_clip_skip = gr.Dropdown(label="CLIP Skip", choices=[0, 1, 2], value=1)
                                        t2i_batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, value=1, step=1)
                                
                                with gr.Row():                          
                                    t2i_controlnet_dropdown = gr.Dropdown(
                                            label="Select Controlnet", 
                                            choices=["None", "Canny - control_v11p_sd15_canny", "Depth - control_v11f1p_sd15_depth","OpenPose - control_v11p_sd15_openpose"], 
                                            value="None"
                                        )
                                    t2i_controlnet_strength = gr.Slider(label="ControlNet Strength", minimum=0, maximum=1, value=1.0, step=0.01, visible=False)
                                t2i_control_image = gr.Image(type="pil", label="Control Image", height=300, visible=False)
                                
                                    
                        t2i_output_image = gr.Gallery(type="pil", label="Generated Image(s)", selected_index=0, columns=1, rows=1, visible=True)
                                       
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            t2i_lora_prompt = gr.Textbox(label="Adjust LoRA Weights", lines=1, visible=False)
                            t2i_prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=8)
                        with gr.Column():
                            with gr.Row():
                                t2i_generate_button = gr.Button("Generate Image")
                                t2i_clear_button = gr.Button("Clear Output")
                            with gr.Row(equal_height=True): 
                                t2i_output_seed = gr.Textbox(value=-1, label="Output Seed", interactive=False, show_copy_button=True,show_label=True, visible=False)   
                                t2i_send_to_i2i_btn = gr.Button("Send To Image To Image Tab", visible=False)
                            t2i_negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here...", lines=8)
                            

                    # Listeners
                    gr.on(
                        triggers=[t2i_base_model_dropdown.change, t2i_checkpoint_dropdown.change, t2i_controlnet_dropdown.change, t2i_scheduler_dropdown.change],
                        fn=button_is_waiting,
                        inputs=None,
                        outputs=t2i_generate_button
                    )
                            
                    gr.on(
                        triggers=[t2i_base_model_dropdown.change,t2i_checkpoint_dropdown.change, t2i_controlnet_dropdown.change, t2i_scheduler_dropdown.change],
                        fn=self.load_t2i,
                        inputs=[t2i_base_model_dropdown, t2i_checkpoint_dropdown, t2i_scheduler_dropdown, t2i_controlnet_dropdown, t2i_use_lora, t2i_lora_prompt],
                        outputs=t2i_generate_button
                    )

                    t2i_reset_seed_btn.click(fn=reset_seed, outputs=t2i_seed)
                    t2i_random_seed_btn.click(fn=random_seed, outputs=t2i_seed)
                    t2i_reuse_seed_btn.click(fn=recycle_seed,inputs=t2i_output_seed, outputs=t2i_seed)
                    
                    t2i_generate_button.click(
                        fn=clear_gallery,
                        inputs=None,
                        outputs=t2i_output_image
                    )
                    
                    t2i_generate_button.click(
                        fn=generating,
                        inputs=None,
                        outputs=t2i_generate_button  
                    )
                    
                    t2i_generate_button.click(
                        fn=make_visible,
                        inputs=None,
                        outputs=[t2i_output_seed]
                    )
                    
                    t2i_generate_button.click(
                        fn=make_visible,
                        inputs=None,
                        outputs=[t2i_output_image]
                    )
                    
                    t2i_generate_button.click(
                        fn=make_visible,
                        inputs=None,
                        outputs=[t2i_send_to_i2i_btn]
                    )
                    
                    t2i_generate_button.click(
                        fn=self.load_and_seed_gen_t2i,
                        inputs=[t2i_load_status, t2i_base_model_dropdown, t2i_checkpoint_dropdown, t2i_scheduler_dropdown, t2i_controlnet_dropdown, t2i_use_lora, t2i_lora_prompt],
                        outputs=[t2i_load_status]
                    )
                                                                   
                    t2i_load_status.change(
                        fn=self.seed_and_gen_t2i,
                         inputs=[t2i_base_model_dropdown, t2i_controlnet_dropdown, t2i_seed, t2i_prompt, t2i_negative_prompt, t2i_width, t2i_height, t2i_steps, t2i_cfg_scale, t2i_clip_skip,t2i_control_image, t2i_batch_size, t2i_controlnet_strength, t2i_hires_fix, t2i_use_lora, t2i_lora_dropdown, t2i_lora_prompt],
                         outputs=[t2i_generate_button, t2i_output_seed, t2i_output_image]
                    )
                    
                    t2i_base_model_dropdown.change(fn=change_controlnet, inputs=t2i_base_model_dropdown,outputs=t2i_controlnet_dropdown)
                    t2i_controlnet_dropdown.change(fn=upload_control_img,inputs=t2i_controlnet_dropdown,outputs=[t2i_control_image,t2i_controlnet_strength])
                    t2i_use_lora.change(fn=using_lora, inputs=t2i_use_lora, outputs=[t2i_lora_dropdown,t2i_lora_prompt])
                    t2i_use_lora.change(fn=lora_to_prompt_cb,inputs=[t2i_use_lora,t2i_lora_dropdown],outputs=t2i_lora_prompt)
                    t2i_lora_dropdown.change(fn=lora_to_prompt, inputs=t2i_lora_dropdown,outputs=t2i_lora_prompt)
                    t2i_clear_button.click(fn=clear, inputs=None, outputs=t2i_output_image)        
                    t2i_send_to_i2i_btn.click(fn=send_to_inpaint,inputs=t2i_output_image,outputs=input_image)  
                
                with gr.TabItem("PNG Info"):
                    with gr.Row():
                        png_input_image = gr.Image(type="pil", label="Input Image", height=600)
                        with gr.Column():
                            png_info = gr.Textbox(label="Generation Parameters",lines=25, show_copy_button=True,show_label=True)
                            with gr.Row(equal_height=True):
                                info_to_i2i_btn = gr.Button("Send Parameters to Image To Image Tab",visible=False)   
                                info_to_t2i_btn = gr.Button("Send Parameters to Text to Image Tab",visible=False)  
                                state= gr.State(value=0)                            
                                               
                    #Listeners
                    png_input_image.change(fn=get_metadata, inputs=png_input_image, outputs=[png_info, info_to_i2i_btn, info_to_t2i_btn])
                    
                    info_to_i2i_btn.click(
                    fn=load_info_to_i2i,
                    inputs=[png_info,state],
                    outputs=[
                        base_model_dropdown,
                        checkpoint_dropdown,
                        scheduler_dropdown,
                        controlnet_dropdown,
                        controlnet_strength,
                        seed,
                        prompt,  
                        negative_prompt,  
                        width,  
                        height,  
                        steps, 
                        cfg_scale,  
                        clip_skip, 
                        batch_size,
                        use_lora,
                        lora_dropdown,
                        lora_prompt,
                        input_image,
                        mask_blur, 
                        fill_setting,
                        mask_crop,
                        maintain_aspect_ratio,
                        post_process,
                        denoise_strength,
                        mode, 
                        outpaint_img_pos,
                        outpaint_max_dim, 
                        state
                    ])
                    
                    
                    
                    info_to_t2i_btn.click(
                    fn=load_info_to_t2i,
                    inputs=png_info,
                    outputs=[
                        t2i_base_model_dropdown,
                        t2i_checkpoint_dropdown,
                        t2i_scheduler_dropdown,
                        t2i_controlnet_dropdown,
                        t2i_controlnet_strength,
                        t2i_seed,
                        t2i_prompt,  
                        t2i_negative_prompt,  
                        t2i_width,  
                        t2i_height,  
                        t2i_steps, 
                        t2i_cfg_scale,  
                        t2i_clip_skip, 
                        t2i_batch_size,
                        t2i_hires_fix,
                        t2i_use_lora,
                        t2i_lora_dropdown,
                        t2i_lora_prompt
                    ])
                    
                    
                    state.change(
                        fn=load_mask_to_inpaint,
                        inputs=png_info,
                        outputs=[inpaint_mask]
                    )                                    

        iface.queue().launch(share= not IS_LOCAL)
        
        
    def load_t2i(self, base_model, checkpoint, scheduler, controlnet, use_lora, lora_dict, generate_button_clicked=False):
        """Load the pipeline based on changes in the checkpoint or ControlNet selection."""
        if generate_button_clicked==False:
            lora_dict = self.parse_lora_prompt(lora_dict)
        if self.is_compatible(base_model,checkpoint):
            # Load pipeline with the determined parameters
            self.pipeline_manager.load_pipeline(base_model, checkpoint, "txt2img", scheduler, controlnet_type=controlnet, use_lora=use_lora, lora_dict=lora_dict) 
            return gr.update(interactive=True, value="Generate Image") if generate_button_clicked is False else gr.update(interactive=False, value="Generating...")     
        else:
            gr.Warning('Base model and checkpoint are not compatible.', duration=5)
            return gr.update(interactive=True, value="Generate Image") 
    
    def load_and_seed_gen_t2i(self, load_status, base_model, checkpoint, scheduler, controlnet, use_lora, lora_prompt):
        """Load models and generate when generate button clicked."""
        
        lora_dict=self.parse_lora_prompt(lora_prompt)
        load_status ^= 1
        self.load_t2i(base_model, checkpoint, scheduler, controlnet,use_lora, lora_dict, generate_button_clicked=True)
        
        return load_status
    
    
    def seed_and_gen_t2i( self, base_model, controlnet_name, seed, *args):
        """Generate an inpainted image after loading the appropriate model."""
        
        if seed is None or seed == "" or seed == -1:
            # Generate a random seed if not provided
            seed = random.randint(0, 2**32 - 1)  # Generate a random seed
            generator = Generator(device=DEVICE).manual_seed(seed)  # Create a generator with the random seed
        else:
            seed = int(seed)
        generator = Generator(device=DEVICE).manual_seed(seed)  # Create a generator with the specified seed
        # Generate the inpainted image with the loaded pipeline
        
        return gr.update(interactive=True, value="Generate Image"), str(seed), generate_t2i_image(
            self.pipeline_manager, base_model, controlnet_name, seed, generator,
            *args) 
            
    def load_i2i(self, base_model, checkpoint, scheduler, controlnet, use_lora, lora_dict, mode, generate_button_clicked=False):
        """Load the pipeline based on changes in the checkpoint or ControlNet selection."""
        if generate_button_clicked==False:
            lora_dict = self.parse_lora_prompt(lora_dict)
        # Load pipeline with the determined parameters
        if self.is_compatible(base_model,checkpoint):
            if mode=="Image To Image":
                self.pipeline_manager.load_pipeline(base_model, checkpoint, "img2img", scheduler, controlnet_type=controlnet, use_lora=use_lora, lora_dict=lora_dict)
            else:
                self.pipeline_manager.load_pipeline(base_model, checkpoint, "inpainting", scheduler, controlnet_type=controlnet, use_lora=use_lora, lora_dict=lora_dict)
            return gr.update(interactive=True, value="Generate Image") if generate_button_clicked is False else gr.update(interactive=False, value="Generating...")
        else:
            gr.Warning('Base model and checkpoint are not compatible.', duration=5)
            return gr.update(interactive=True, value="Generate Image")
        
    def load_and_seed_gen_i2i(self, load_status, base_model, checkpoint, scheduler, controlnet, use_lora, lora_dict, mode):
        """Load models and generate when generate button clicked."""
        
        lora_dict=self.parse_lora_prompt(lora_dict)
        load_status ^= 1
        self.load_i2i(base_model, checkpoint, scheduler, controlnet, use_lora, lora_dict, mode, generate_button_clicked=True)
        
        return load_status
    
    
    def seed_and_gen_i2i(self, base_model, controlnet_name, seed, input_image, *args):
        """Generate an inpainted image after loading the appropriate model."""
        
        if input_image is None:
            return gr.update(interactive=True, value="Generate Image"),str(seed), []

        if seed is None or seed == "" or seed == -1:
            # Generate a random seed if not provided
            seed = random.randint(0, 2**32 - 1)  # Generate a random seed
            generator = Generator(device=DEVICE).manual_seed(seed)  # Create a generator with the random seed
        else:
            seed = int(seed)
        generator = Generator(device=DEVICE).manual_seed(seed)  # Create a generator with the specified seed
        # Generate the inpainted image with the loaded pipeline
        
        return gr.update(interactive=True, value="Generate Image"), str(seed), generate_image(
            self.pipeline_manager, base_model, controlnet_name, seed, input_image, generator,
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
    
    def is_compatible(self, base_model: str, checkpoint: str) -> bool:
        # Convert both base_model and checkpoint to lowercase
        base_model_lower = base_model.lower()
        checkpoint_lower = checkpoint.lower()
        
        # Check if both contain "xl" or neither contain "xl"
        if ("xl" in base_model_lower and "xl" in checkpoint_lower) or \
        ("xl" not in base_model_lower and "xl" not in checkpoint_lower):
            return True
        else:
            return False

# Main execution
if __name__ == "__main__":
    
    app = StableDiffusionApp()

