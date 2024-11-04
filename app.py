# app.py

import gradio as gr
from modules.inpainting import *
# from modules.txt2img import generate_txt2img_image
from modules.upscale import upscale
from modules.manage_models import manage_models_tab, model_dir
from modules.pipelines import PipelineManager
from torch import Generator
import random   

class StableDiffusionApp:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.pipeline_manager = PipelineManager(model_dir)
        # Load the default inpainting pipeline
        self.pipeline_manager.load_pipeline()
        self.setup_gradio_interface()

    def setup_gradio_interface(self):
        """Create the Gradio interface with tabs for inpainting, text-to-image, and more."""
               
        with gr.Blocks(
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
            button[aria-label="Clear"] { 
                display: none;  /* Hides the Clear button */ 
            }
            """,
            theme=gr.themes.Default(primary_hue="green", secondary_hue="pink")
        ) as iface:

            # Create the tabs
            with gr.Tabs():
                # Inpainting Tab
                with gr.TabItem("Inpainting"):
                    brush = gr.Brush(colors=["#000000"], color_mode='fixed',default_size=50)
                    
                    with gr.Row():
                        inpaint_input_image = gr.Image(type="pil", label="Input Image", height=600)
                        inpaint_mask = gr.ImageEditor(type="pil", label="Mask Editor", height=600, brush=brush, transforms=(), sources=('clipboard'), placeholder="Mask Preview", layers=False)
                        output_image = gr.Gallery(type="pil", label="Generated Image(s)", height=600, selected_index=0, columns=1, rows=1, visible=False)

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            custom_dimensions = gr.Checkbox(label="Custom Output Dimensions", value=False)
                            with gr.Row():
                                width = gr.Dropdown(label="Width", choices=[512, 768, 1024], value=512, visible=False, allow_custom_value=True)
                                height = gr.Dropdown(label="Height", choices=[512, 768, 1024], value=768, visible=False, allow_custom_value=True)
                            mode = gr.Radio(["Inpaint", "Outpaint"], value="Inpaint", label = "Mode")
                            mask_blur = gr.Slider(label="Mask Blur", minimum=0, maximum=40, value=0, step=0.1)

                        with gr.Column(scale=1):
                            fill_setting = gr.Radio(label="Mask", choices=["Generate Inside Mask", "Generate Outside Mask"], value="Generate Inside Mask")
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
                            batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, value=1, step=1)

                        with gr.Column(scale=1):
                            generate_button = gr.Button(value="Generate Image")
                            clear_button = gr.Button("Clear Image")
                            upscalefour = gr.Button("Upscale to 4x")
                            output_seed = gr.Textbox(value=None, label="Output Seed", interactive=False, show_copy_button=True, visible=False)

                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=6)
                        negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here...", lines=6)

                    with gr.Row(equal_height=True):
                        with gr.Group():
                            seed = gr.Number(label="Seed", value=-1)
                            #generator = gr.State(Generator().manual_seed(random.randint(0, 2**32 - 1)))  # Initial generator state
                            with gr.Column():
                                reset_seed_btn = gr.Button("🔄", size='lg')
                                random_seed_btn = gr.Button("🎲", size='lg')
                        steps = gr.Number(label="Number of Steps", value=25)
                        cfg_scale = gr.Number(label="CFG Scale", value=7)
                        clip_skip = gr.Dropdown(label="CLIP Skip", choices=[0, 1, 2, 3], value=0)

                    with gr.Row():
                        # Dropdown for selecting inpainting checkpoint
                        inpainting_checkpoint_dropdown = gr.Dropdown(
                            label="Select Inpaint Checkpoint", 
                            choices=["runwayml/stable-diffusion-inpainting","stablediffusionapi/realistic-vision-v6.0-b1-inpaint", "runwayml/stable-diffusion-v1-5", "sd-v1-5-inpainting.ckpt"], 
                            value="stablediffusionapi/realistic-vision-v6.0-b1-inpaint",
                        )
                            
                        # Dropdown for selecting scheduler
                        scheduler_dropdown = gr.Dropdown(
                            label="Select Scheduler", 
                            choices=["DPMSolverMultistepScheduler", "DDIMScheduler", "EulerAncestralDiscreteScheduler"], 
                            value="DPMSolverMultistepScheduler",
                        )
                        
                    toggle_mode_hide_components = [custom_dimensions, fill_setting, maintain_aspect_ratio]
                    component_hide_count = len(toggle_mode_hide_components)
                    
                    toggle_mode_show_components = [outpaint_img_pos,outpaint_max_dim]
                    component_show_count = len(toggle_mode_show_components)
                    

                    # def clear_image():
                    #     return None
                    
                    def clear_gallery():
                        return []

                    def make_visible():
                        return gr.update(visible=True)
                                            
                    def button_is_waiting():
                        return gr.update(interactive=False, value="Loading...")

                    def toggle_dimensions(custom_dim):
                        if not custom_dim:
                            return gr.update(visible=False), gr.update(visible=False)  # Hide width and height
                        else:
                            return gr.update(visible=True), gr.update(visible=True)  # Show width and height
                        
                    def toggle_mode(mode):
                        if mode=="Inpaint":
                            return gr.update(brush=brush, transforms=())
                        if mode=="Outpaint":
                            return gr.update(brush=False, transforms=('crop'))
                        
                    def toggle_mode_hide(mode):
                        # Determine the visibility state based on the mode
                        visible = (mode == "Inpaint")  # True if Inpaint 
                        updates = [gr.update(visible=visible) for _ in range(component_hide_count)] 
                        return updates  # Return the list of updates
                    
                    def toggle_mode_show(mode):
                        visible = (mode == "Outpaint")  # True if Outpaint
                        updates = [gr.update(visible=visible) for _ in range(component_show_count)] 
                        return updates  # Return the list of updates

                    def reset_seed():
                        return -1

                    def random_seed():
                        return random.randint(0, 2**32 - 1)

                    # Listen for events
                    inpaint_input_image.change(fn=retrieve_mask, inputs=[mode, outpaint_img_pos, inpaint_input_image], outputs=[inpaint_mask])
                    outpaint_img_pos.change(fn=retrieve_mask, inputs=[mode, outpaint_img_pos, inpaint_input_image], outputs=[inpaint_mask])
                    mode.change(fn=retrieve_mask, inputs=[mode, outpaint_img_pos, inpaint_input_image], outputs=[inpaint_mask])
                    mode.change(fn=toggle_mode,inputs=mode,outputs=inpaint_mask)
                    mode.change(fn=toggle_mode_hide,inputs=mode, outputs=toggle_mode_hide_components)
                    mode.change(fn=toggle_mode_show,inputs=mode, outputs=toggle_mode_show_components)
                           
                    gr.on(
                        triggers=[inpainting_checkpoint_dropdown.change, controlnet_dropdown.change],
                        fn=button_is_waiting,
                        inputs=None,
                        outputs=generate_button
                    )
                            
                    gr.on(
                        triggers=[inpainting_checkpoint_dropdown.change, controlnet_dropdown.change, scheduler_dropdown.change],
                        fn=self.load_inpaint,
                        inputs=[inpainting_checkpoint_dropdown, scheduler_dropdown, controlnet_dropdown],
                        outputs=generate_button
                    )

                    reset_seed_btn.click(fn=reset_seed, outputs=seed)
                    random_seed_btn.click(fn=random_seed, outputs=seed)

                    generate_button.click(
                        fn=clear_gallery,
                        inputs=None,
                        outputs=output_image
                    )

                    generate_button.click(
                        fn=make_visible,
                        inputs=None,
                        outputs=[output_seed]
                    )

                    generate_button.click(
                        fn=make_visible,
                        inputs=None,
                        outputs=[output_image]
                    )
                    
                    generate_button.click(
                        fn=self.seed_and_gen_inpaint_image,
                        inputs=[seed, prompt, negative_prompt, width, height, steps, cfg_scale, clip_skip, inpaint_mask, fill_setting, inpaint_input_image, maintain_aspect_ratio, post_process, custom_dimensions, denoise_strength, batch_size, mask_blur, mode, outpaint_img_pos, outpaint_max_dim],
                        outputs=[output_seed, output_image]
                    )
                    
                    clear_button.click(fn=lambda: None, inputs=[], outputs=output_image)
                    upscalefour.click(fn=upscale, inputs=output_image, outputs=output_image)

                    custom_dimensions.change(fn=toggle_dimensions, inputs=[custom_dimensions], outputs=[width, height])

                # # Text-to-Image Tab
                # with gr.TabItem("Text-to-Image"):
                #     with gr.Row():
                #         txt2img_prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=4)
                #         txt2img_negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here...", lines=4)

                #     with gr.Row():
                #         txt2img_width = gr.Dropdown(label="Width", choices=[512, 768, 1024], value=512)
                #         txt2img_height = gr.Dropdown(label="Height", choices=[512, 768, 1024], value=768)
                #         txt2img_seed = gr.Number(label="Seed", value=-1)
                #         txt2img_steps = gr.Number(label="Number of Steps", value=25)
                #         txt2img_cfg_scale = gr.Number(label="CFG Scale", value=7)
                #         txt2img_clip_skip = gr.Dropdown(label="CLIP Skip", choices=[0, 1, 2, 3], value=0)

                #     # Dropdown for selecting text-to-image model
                #     txt2img_model_dropdown = gr.Dropdown(
                #         label="Select Text-to-Image Model", 
                #         choices=["runwayml/stable-diffusion-v1-5"], 
                #         value="runwayml/stable-diffusion-v1-5",
                #     )

                #     txt2img_generate_button = gr.Button("Generate Image")
                #     txt2img_output_image = gr.Image(type="pil", label="Generated Image")

                #     txt2img_clear_button = gr.Button("Clear Image")
                #     txt2img_clear_button.click(fn=lambda: None, inputs=[], outputs=txt2img_output_image)

                #     txt2img_generate_button.click(
                #         fn=self.generate_and_load_txt2img_image,
                #         inputs=[txt2img_model_dropdown, txt2img_prompt, txt2img_negative_prompt, txt2img_width, txt2img_height, txt2img_seed, txt2img_steps, txt2img_cfg_scale, txt2img_clip_skip],
                #         outputs=txt2img_output_image
                #     )


                                
                with gr.TabItem("PNG Info"):
                    with gr.Row(equal_height=True):
                        png_input_image = gr.Image(type="pil", label="Input Image")
                        with gr.Column():
                            png_info = gr.Textbox(label="Generation Parameters",lines=6)
                            info_to_inpaint_btn = gr.Button("Send Parameters to Inpaint Tab",visible=False)
                        
                        
                    def get_metadata(image):
                        if image is None:
                            return "No metadata available."
                        
                        # Extract metadata
                        metadata = image.info  # Metadata dictionary for PNG images in PIL
                        if not metadata:
                            return "No metadata found in this image."

                        # Format metadata for display
                        return "\n".join([f"{k}: {v}" for k, v in metadata.items()])
                    
                    def load_info_to_inpaint(info):
                        # Split the input into lines
                        lines = info.strip().split('\n')
                        
                        # Create a dictionary to hold the parsed parameters
                        parameters = {}
                        
                        # Iterate through each line and extract key-value pairs
                        for line in lines:
                            if ": " in line:
                                key, value = line.split(": ", 1)  # Split only on the first occurrence
                                parameters[key.strip()] = value.strip()

                        # Return updates for each component based on the parsed parameters
                        return (
                            parameters.get("model/checkpoint","stablediffusionapi/realistic-vision-v6.0-b1-inpaint"),
                            parameters.get("scheduler","DPMSolverMultistepScheduler"),
                            parameters.get("controlnet", "None"),
                            parameters.get("seed", -1),            
                            parameters.get("prompt", ""),             
                            parameters.get("negative_prompt", ""),        
                            int(parameters.get("width", 512)),              
                            int(parameters.get("height", 768)),            
                            int(parameters.get("steps", 25)),   
                            float(parameters.get("mask_blur", 0)),       
                            float(parameters.get("cfg_scale", 7.0)),     
                            int(parameters.get("clip_skip", 0)),             
                            parameters.get("fill_setting", "Generate Inside Mask"),  
                            parameters.get("maintain_aspect_ratio", "True") == "True", 
                            parameters.get("post_process", "True") == "True",  
                            parameters.get("custom_dimensions", "True") == "True",  
                            float(parameters.get("denoise_strength", 1.0)),  
                            int(parameters.get("batch_size", 1)),
                            parameters.get("mode","Inpaint"),
                            parameters.get("image_positioned_at","Center"),
                            parameters.get("maximum_width/height", 768),        
                        )
                        
                        
                        
                    #Listeners
                    png_input_image.change(fn=make_visible,inputs=None,outputs=info_to_inpaint_btn)
                    
                    png_input_image.change(fn=get_metadata, inputs=png_input_image, outputs=png_info)
                    info_to_inpaint_btn.click(
                    fn=load_info_to_inpaint,
                    inputs=png_info,
                    outputs=[
                        inpainting_checkpoint_dropdown,
                        scheduler_dropdown,
                        seed,  
                        prompt,  
                        negative_prompt,  
                        width,  
                        height,  
                        steps, 
                        cfg_scale,  
                        clip_skip, 
                        fill_setting, 
                        maintain_aspect_ratio,  
                        post_process,  
                        custom_dimensions, 
                        denoise_strength, 
                        batch_size 
                    ])
                    
                # Image Upscale Tab
                with gr.TabItem("Image Upscale"):
                    with gr.Row():
                        upscale_input_image = gr.Image(type="pil", label="Input Image", scale=2)
                        upscale_output_image = gr.Image(type="pil", label="Output Image", scale=4)  # Make this container larger
                    with gr.Row(equal_height=True):
                        upscale_factor = gr.Radio(label="Upscale Factor", choices=["2", "4"], value="4", type="value")
                        with gr.Column():
                            upscale_button = gr.Button("Upscale")
                            upscale_to_inpaint = gr.Button("Send to Inpaint Tab")
                    
                    def send(img):
                        return img
            
                    #Listeners   
                    upscale_button.click(
                        fn=upscale,
                        inputs=[upscale_input_image, upscale_factor],
                        outputs=upscale_output_image
                    )             
                    upscale_button.click(fn=upscale, inputs=[upscale_input_image, upscale_factor], outputs=upscale_output_image)
                    upscale_to_inpaint.click(fn=send,inputs=upscale_output_image, outputs=inpaint_input_image)
             
                #manage_models_tab()

        iface.queue().launch(share= not is_local)
         
    def load_inpaint(self, checkpoint, scheduler, controlnet):
        """Load the pipeline based on changes in the checkpoint or ControlNet selection."""

        # Load pipeline with the determined parameters
        self.pipeline_manager.load_pipeline(checkpoint, "inpainting", scheduler, controlnet_type=controlnet)
        return gr.update(interactive=True, value="Generate Image")
    
    def seed_and_gen_inpaint_image(self, seed, *args):
        """Generate an inpainted image after loading the appropriate model."""
          
        if seed is None or seed == "" or seed == -1:
            # Generate a random seed if not provided
            seed = random.randint(0, 2**32 - 1)  # Generate a random seed
            generator = Generator().manual_seed(seed)  # Create a generator with the random seed
        else:
            seed = int(seed)
        generator = Generator().manual_seed(seed)  # Create a generator with the specified seed
        # Generate the inpainted image with the loaded pipeline
        
        return str(seed), generate_inpaint_image(
            self.pipeline_manager, seed, generator,
            *args) 
               
    # def generate_and_load_txt2img_image(self, selected_checkpoint, *args):
    #     """Generate a txt2img image after loading the appropriate checkpoint."""
    #     self.pipeline_manager.lo+ad_pipeline(selected_checkpoint, 'txt2img')
    #     # Generate the txt2img image with the loaded pipeline
    #     return generate_txt2img_image(self.pipeline_manager, *args)

        
# Main execution
if __name__ == "__main__":
    is_local = os.getenv("MYAPP_DEV_ENV") == "true"
    app = StableDiffusionApp(model_dir)

