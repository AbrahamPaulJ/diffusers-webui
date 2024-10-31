# app.py

import gradio as gr
from modules.inpainting import *
from modules.txt2img import generate_txt2img_image
from modules.upscale import upscale
from modules.manage_models import manage_models_tab, model_dir
from pipelines import PipelineManager
from torch import Generator
import random

class StableDiffusionApp:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.pipeline_manager = PipelineManager(model_dir)

        # Load the default inpainting pipeline
        self.pipeline_manager.load_pipeline() #defaults are specified in pipelines.py
        self.setup_gradio_interface()
        
    def setup_gradio_interface(self):
        """Create the Gradio interface with tabs for inpainting and text-to-image."""
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
        button[aria-label="Transform button"] { 
            display: none;  /* Hides the Transform button */ 
        }
        button[aria-label="Clear"] { 
            display: none;  /* Hides the Clear button */ 
        }
        """,
        theme=gr.themes.Default(primary_hue="green", secondary_hue="pink")
        ) as iface:

            # gr.Markdown("### Stable Diffusion")
            
            # Create the tabs
            with gr.Tabs():
                self.create_inpainting_tab()
                self.create_txt2img_tab()
                self.create_upscaletab()
                manage_models_tab() 

        iface.launch()
        
        
    def create_inpainting_tab(self):
        """Define the UI and functionality for the Inpainting tab."""
        with gr.TabItem("Inpainting"):
            brush = gr.Brush(colors=["#000000"], color_mode='fixed')
            with gr.Row():
                    input_image = gr.Image(type="pil", label="Input Image", height=600)
                    mask_output = gr.ImageEditor(type="pil", label="Mask Preview", height=600, brush=brush)
                    output_image = gr.Gallery(type="pil",label="Generated Image(s)", height=600, selected_index=0,columns=1,rows=1)
                
            with gr.Row(equal_height=True):
                
                with gr.Column(scale=1):
                    custom_dimensions = gr.Checkbox(label="Custom Dimensions", value=False)
                    width = gr.Dropdown(label="Width", choices=[512, 768, 1024], value=512, visible=False)
                    height = gr.Dropdown(label="Height", choices=[512, 768, 1024], value=768, visible=False)
                    
                with gr.Column(scale=1):
                    segment_type = gr.Radio(label="Inpaint Helper", choices=["Use Brush"], value="Use Brush")
                    denoise_strength = gr.Slider(label="Denoise Strength", minimum=0, maximum=1, value=1, step=0.01)
                    
                with gr.Column(scale=1):
                    maintain_aspect_ratio = gr.Checkbox(label="Maintain Aspect Ratio (Auto Padding)", value=True)
                    post_process = gr.Checkbox(label="Post-Processing", value=True)
                    use_controlnet = gr.Checkbox(label="ControlNet", value=False)
                    batch_size= gr.Slider(label="Batch Size", minimum=1, maximum=8, value=1, step=1)
                    
                with gr.Column(scale=1):
                    generate_button = gr.Button("Generate Image")
                    clear_button = gr.Button("Clear Image")
                    upscalefour = gr.Button("Upscale to 4x")
                    output_seed = gr.Textbox(value=None, label="Output Seed", interactive=False,show_copy_button=True, visible=False)
                    
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=6)
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here...", lines=6)
                        
            with gr.Row(equal_height=True):

                seed = gr.Number(label="Seed", value=-1)
                steps = gr.Number(label="Number of Steps", value=25)
                cfg_scale = gr.Number(label="CFG Scale", value=7)
                clip_skip = gr.Dropdown(label="CLIP Skip", choices=[0, 1, 2, 3], value=0)
                with gr.Column(scale=1):
                    fill_setting = gr.Radio(label="Mask", choices=["Generate Inside Mask", "Generate Outside Mask"], value="Generate Inside Mask")

            with gr.Row():
                
                # Dropdown for selecting inpainting model
                inpainting_model_dropdown = gr.Dropdown(
                    label="Select Inpaint Checkpoint", 
                    choices=["stablediffusionapi/realistic-vision-v6.0-b1-inpaint","runwayml/stable-diffusion-v1-5","sd-v1-5-inpainting.ckpt"], 
                    value="stablediffusionapi/realistic-vision-v6.0-b1-inpaint",
                )
                 # Dropdown for selecting scheduler
                scheduler_dropdown = gr.Dropdown(
                    label="Select Scheduler", 
                    choices=["DPMSolverMultistepScheduler", "DDIMScheduler", "EulerAncestralDiscreteScheduler"], 
                    value="DPMSolverMultistepScheduler",
                )
                               
                none_state = gr.State(value=None) 
                
                def clear_gallery():
                    return []

                    
                def process_segmentation(segment_type, image):
                    if segment_type == "Use Brush":
                        return reset_brush(image)  
                    
                def make_visible():
                    return gr.update(visible=True)
                    
                def toggle_dimensions(custom_dim):
                    if not custom_dim:
                        return gr.update(visible=False), gr.update(visible=False)  # Hide width and height
                    else:
                        return gr.update(visible=True), gr.update(visible=True)  # Show width and height
               
            # Listen for events
            input_image.change(fn=reset_brush, inputs=input_image,outputs=mask_output)
            
            segment_type.change(
            fn=process_segmentation,  # Call the processing function
            inputs=[segment_type, input_image],  # Get both the selected type and the input image
            outputs=mask_output         # Output to display the mask
            )
                       
            generate_button.click(
            fn=make_visible,
            inputs=None,
            outputs=output_seed)
            
            generate_button.click(
            fn=clear_gallery,
            inputs=None,
            outputs=output_image)
            
            generate_button.click(
            fn=self.seed_and_gen_inpaint_image,
            inputs=[inpainting_model_dropdown, scheduler_dropdown, use_controlnet, seed, prompt, negative_prompt, width, height, steps, cfg_scale, clip_skip, mask_output, fill_setting, input_image, segment_type, maintain_aspect_ratio, post_process, custom_dimensions, denoise_strength, batch_size],
            outputs=[output_seed,output_image])
            
            inpainting_model_dropdown.change(fn= self.load_inpaint, inputs=[inpainting_model_dropdown, none_state, none_state], outputs=None)
            scheduler_dropdown.change(fn= self.load_inpaint, inputs=[none_state,scheduler_dropdown, none_state], outputs=None)
            use_controlnet.change(fn= self.load_inpaint, inputs=[none_state, none_state, use_controlnet], outputs=None)
            
            clear_button.click(fn=lambda: None, inputs=[], outputs=output_image)
            upscalefour.click(fn=upscale,inputs=output_image, outputs = output_image)
            
            custom_dimensions.change(fn=toggle_dimensions, inputs=[custom_dimensions], outputs=[width, height])
            
    def create_txt2img_tab(self):
        """Define the UI and functionality for the Text-to-Image tab."""
        with gr.TabItem("Text-to-Image"):
            with gr.Row():
                txt2img_prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=4)
                txt2img_negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here...", lines=4)

            with gr.Row():
                txt2img_width = gr.Dropdown(label="Width", choices=[512, 768, 1024], value=512)
                txt2img_height = gr.Dropdown(label="Height", choices=[512, 768, 1024], value=768)
                txt2img_seed = gr.Number(label="Seed", value=-1)
                txt2img_steps = gr.Number(label="Number of Steps", value=25)
                txt2img_cfg_scale = gr.Number(label="CFG Scale", value=7)
                txt2img_clip_skip = gr.Dropdown(label="CLIP Skip", choices=[0, 1, 2, 3], value=0)

            # Dropdown for selecting text-to-image model
            txt2img_model_dropdown = gr.Dropdown(
                label="Select Text-to-Image Model", 
                choices=["runwayml/stable-diffusion-v1-5"], 
                value="runwayml/stable-diffusion-v1-5",
            )
            
            txt2img_generate_button = gr.Button("Generate Image")
            txt2img_output_image = gr.Image(type="pil", label="Generated Image")
       
            clear_button = gr.Button("Clear Image")
            clear_button.click(fn=lambda: None, inputs=[], outputs=txt2img_output_image)

            txt2img_generate_button.click(
                fn=self.generate_and_load_txt2img_image,
                inputs=[txt2img_model_dropdown, txt2img_prompt, txt2img_negative_prompt, txt2img_width, txt2img_height, txt2img_seed, txt2img_steps, txt2img_cfg_scale, txt2img_clip_skip],
                outputs=txt2img_output_image
            )
            
    def create_upscaletab(self):
        """Define the UI and functionality for the Upscale tab."""
        with gr.TabItem("Image Upscale"):
            with gr.Row():
                upscale_input_image = gr.Image(type="pil", label="Input Image", elem_id="input_image", scale=2)
                upscale_output_image = gr.Image(type="pil", label="Output Image", elem_id="output_image", scale=4)  # Make this container larger
            with gr.Row():
                upscale_factor = gr.Radio(label="Upscale Factor", choices=["2", "4"], value="4", type="value")
                upscale_button = gr.Button("Upscale")
                  
            upscale_button.click(fn=upscale, inputs=[upscale_input_image, upscale_factor], outputs=upscale_output_image)
            
    def seed_and_gen_inpaint_image(self, selected_model, scheduler, use_controlnet, seed, *args):
        """Generate an inpainted image after loading the appropriate model."""
        # Reload pipeline if necessary
        self.pipeline_manager.load_pipeline(selected_model, 'inpainting', scheduler, use_controlnet=use_controlnet)
        
        if seed is None or seed == "" or seed == -1:
            # Generate a random seed if not provided
            seed = random.randint(0, 2**32 - 1)  # Generate a random seed
            generator = Generator().manual_seed(seed)  # Create a generator with the random seed
        else:
            seed = int(seed)
        generator = Generator().manual_seed(seed)  # Create a generator with the specified seed
        # Generate the inpainted image with the loaded pipeline
        
        return str(seed), generate_inpaint_image(
            self.pipeline_manager,scheduler, use_controlnet, generator,
            *args) 
        
    def load_inpaint(self, selected_model=None, scheduler=None, use_controlnet=None):
        """Load the pipeline based on changes in the model or ControlNet selection."""
        
        # Determine the parameters for loading the pipeline
        model_name = selected_model if selected_model is not None else self.pipeline_manager.active_checkpoint
        active_scheduler = scheduler if scheduler is not None else self.pipeline_manager.active_scheduler
        control_net_enabled = use_controlnet if use_controlnet is not None else self.pipeline_manager.control_net_enabled

        # Load the pipeline with the determined parameters
        self.pipeline_manager.load_pipeline(model_name, "inpainting", scheduler=active_scheduler, use_controlnet=control_net_enabled)

        return []

    def generate_and_load_txt2img_image(self, selected_model, *args):
        """Generate a txt2img image after loading the appropriate model."""
        self.pipeline_manager.load_pipeline(selected_model, 'txt2img')
        # Generate the txt2img image with the loaded pipeline
        return generate_txt2img_image(self.pipeline_manager, *args)
        
# Main execution
if __name__ == "__main__":
    app = StableDiffusionApp(model_dir)
