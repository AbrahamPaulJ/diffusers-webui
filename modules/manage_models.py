# modules/manage_modules.py

import os
import gradio as gr
import shutil  # For removing non-empty directories

# Define the custom model directory
rel_model_dir = "models"
model_dir = os.path.abspath(rel_model_dir)

# Function to list model folders and specific file types in the directory
def list_model_folders():
    # Return directories starting with "models" or files ending with ".safetensors" or ".ckpt"
    return [
        name for name in os.listdir(model_dir)
        if (os.path.isdir(os.path.join(model_dir, name)) and name.startswith("models"))
        or (os.path.isfile(os.path.join(model_dir, name)) and (name.endswith(".safetensors") or name.endswith(".ckpt")))
    ]

# Function to delete a selected model
def delete_model(model_name):
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        # Remove model directory or file
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)  # For non-empty directories
        else:
            os.remove(model_path)
        return f"{model_name} deleted successfully."
    else:
        return f"{model_name} not found."

# Function to define the Gradio UI for model management
def manage_models_tab():
    with gr.Tab("Manage Models"):
        # Display list of model folders and files
        model_list = gr.Dropdown(label="Select a model", choices=list_model_folders())
        delete_button = gr.Button("Delete Model")
        output_text = gr.Textbox(label="Status")
        
        delete_button.click(fn=delete_model, inputs=model_list, outputs=output_text)


