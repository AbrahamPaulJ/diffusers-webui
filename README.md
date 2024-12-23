# Installation

Create and activate a virtual environment:
```markdown
python -m venv myenv

source myenv/bin/activate  # On Linux/macOS

myenv\Scripts\activate     # On Windows
```

Clone the repository:
```markdown
git clone https://github.com/AbrahamPaulJ/diffusers-webui.git

cd diffusers-webui
```

# Install dependencies

```markdown
python install.py
```

# Run

```markdown
python app.py
```

# Features

- Supports SD 1.5 and SDXL models.

- Image To Image tab: Allows image-to-image, inpainting and outpainting with the APIs:
[StableDiffusionImg2ImgPipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/img2img)
and [StableDiffusionInpaintPipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/inpaint)

    - Implemented features:

        - ControlNet support with [StableDiffusionControlNetPipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/controlnet) : Canny, Depth and Openpose. 

        - Support for multiple LoRA and embedding files (textual inversion).

        - Inpaint Mode:

            - Manual Brush Tool: Allows users to create a image mask (inpaint mode).
            
            - Gaussian Blur Slider: For applying blur to the mask edges.
            
            - Post-Processing Option: Retains original non-masked areas after image generation. 
        

        - Outpaint Mode: Outpainting is done using the inpaint pipeline, where the a black background canvas is used as a mask.       
              
            - The transform tool allows cropping of the image placed on the black canvas, 
            which makes outpainting in any direction easy.

            - You can choose where to place the image on the canvas using the 
            "Image Positioned at:" option.

            - You can control output image size by adjusting the "Maximum Width/Height" parameter.
        

- Text to Image tab: Generates images from prompts using the API:
[StableDiffusionPipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img).

    - Implemented features:

        - Hi-Res .fix - 2x latent

        - Support for multiple LoRA and embedding files (textual inversion).

<!-- - Image Upscale tab: Includes ESRGAN upscaling options. -->

- PNG Info tab: For previously generated images, featuring a view of the generation
 parameters and options to send the parameters to other tabs.

# Images

![Screenshot](images/txt2img.png)

![Screenshot](images/inpaint.png)

![Screenshot](images/outpaint.png)

# Notes

- You can add new checkpoints/models to the models folder. Supports ".ckpt", ".safetensors" and diffusers formats. 

- Similarly, there are folders for LoRAs and embeddings.

- Prompt format: To add prompt weighting, please use [Compel Prompt Syntax](https://github.com/damian0815/compel/blob/main/Reference.md)

- Ensure you have the correct version of Pytorch and CUDA supported by your device, in case you want to enable GPU for inference.

- To increase generation speed with GPU, you can install xformers if it is supported. It will be detected automatically once installed.

# Known Issues

- Controlnet Strength Slider - doesn't work (diffusers bug).
