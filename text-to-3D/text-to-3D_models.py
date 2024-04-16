import base64
import json
import os
import random
from typing import Union
import diffusers
import numpy as np
import torch
from PIL import Image
import diskcache
from openai import OpenAI
import openai

# Todo: Integrate threestudio into this file
text2img3d_model = {
    "ldm3d"             : ("LDM3D", "Intel/ldm3d"),
    "ldm3d_4c"          : ("LDM3D_4C","Intel/ldm3d-4c"),
    "ldm3d_sr"          : ("LDM3D_SR", [ "Intel/ldm3d-4c","Intel/ldm3d-sr"])
    "ShapE"             : ("ShapE", "openai/shap-e")
    "ProlificDreamer"   : "threestudio"
    'DreamFusion'       : "threestudio"
        'Magic3D'       : "threestudio"
            'SJC'       : "threestudio"
    'Latent-NeRF '      : "threestudio"
    'Fantasia3D'        : "threestudio"
    'TextMesh'          : "threestudio"
    'Zero-1-to-3 '      : "threestudio"
    'Magic123'          : "threestudio"
    'HiFA'              : "threestudio"
    'InstructNeRF2NeRF' : "threestudio"
    'Control4D'         : "threestudio"
    
}



def set_model_key(model_name, key):
	text2img3d_model[model_name] = (text2img3d_model[model_name][0], key)


def list_textto3d_models():
	return list(text2img3d_model.keys())

class AbstractModel:
    def text2img3d(self, prompt):
        "(Abstract method) abstract text2img3d method"

class Text2Img3d:
    def __init__(self, model_name: str = "ldm3d", model: AbstractModel = None, ckpt: str="stabilityai/stable-diffusion-2-1" , precision: torch.dtype = torch.float16, torch_device: str = "cuda", calculate_metrics: bool = False):
        self.model_name = model_name
        self.model = model
        self.ckpt = ckpt
        if isinstance(torch_device, str):
            torch_device = torch.device(torch_device)
        else:
            if torch_device == -1:
                torch_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
            else:
                torch_device = torch.device(f"cuda:{torch_device}")
        if model is None:
            print(f"Loading {model_name} ...")
            class_name, ckpt = text2img3d_model[model_name]
            self.model_presision = precision
            self.model = eval(class_name)(ckpt, precision, torch_device)
            print(f"Finish loading {model_name}")
        else:
            print(f"Using provided model ...")
            class_name = model.__class__.__name__
            self.model = eval(class_name)(ckpt, precision, torch_device)
        self.calculate_metrics = calculate_metrics

    @torch.no_grad()
    def text2img3d(self, prompt):
        img = self.model.text2img3d(prompt)

        return img
    

class LDM3D(AbstractModel):
    def __init__(self, ckpt: str = 'Intel/ldm3d', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionLDM3DPipeline
        self.pipeline = StableDiffusionLDM3DPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
    def text2img(self, prompt):
        output = self.pipeline(prompt)
        return output.rgb[0], output.depth[0]


class LDM3D_4C(AbstractModel):
    def __init__(self, ckpt: str = "Intel/ldm3d-4c", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionLDM3DPipeline
        self.pipeline = StableDiffusionLDM3DPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
    def text2img(self, prompt):
        output = self.pipeline(prompt)
        output_rgb, output_depth = output.rgb, output.depth
        return output_rgb[0], output_depth[0]
    
class LDM3D_Pano(AbstractModel):
    def __init__(self, ckpt: str = "Intel/ldm3d-pano", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionLDM3DPipeline
        self.pipeline = StableDiffusionLDM3DPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
    def text2img(self, prompt):
        output = self.pipeline(prompt,width=1024,
        height=512,
        guidance_scale=7.0,
        num_inference_steps=50,)
        output_rgb, output_depth = output.rgb, output.depth
        return output_rgb[0], output_depth[0]
    
class LDM3D_SR(AbstractModel):
    def __init__(self, ckpt: list =[ "Intel/ldm3d-4c","Intel/ldm3d-sr"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionLDM3DPipeline, DiffusionPipeline
        self.pipeline_ldm3d = StableDiffusionLDM3DPipeline.from_pretrained(ckpt[0], torch_dtype=precision).to(device)
        self.pipeline_upscale = DiffusionPipeline.from_pretrained(ckpt[1], custom_pipeline="pipeline_stable_diffusion_upscale_ldm3d",torch_dtype=precision).to(device)
    def text2img(self, prompt):
        output = self.pipeline_ldm3d(prompt)
        output_rgb, output_depth = output.rgb[0].convert("RGB"), output.depth[0].convert("L")
        outputs = self.pipeline_upscale(prompt="high quality high resolution uhd 4k image", rgb=output_rgb, depth=output_depth, num_inference_steps=50, target_res=[1024, 1024])
        output_rgb, output_depth = outputs.rgb[0], outputs.depth[0]
        return output_rgb, output_depth
    
    
    
class ShapE(AbstractModel):
    def __init__(self, ckpt: str = "openai/shap-e", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import DiffusionPipeline
        self.pipeline = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
    def text2img(self, prompt):
        images = self.pipeline(prompt, guidance_scale=15.0, num_inference_steps=64, frame_size=256).images
        return images[0]

