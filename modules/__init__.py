import os
import diffusers.schedulers 

try:
  import google.colab # type: ignore
  IN_COLAB = True
except:
  IN_COLAB = False
  
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
