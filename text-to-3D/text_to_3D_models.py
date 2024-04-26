import base64
import json
import os
import random
from typing import Union
import diffusers
import numpy as np
import torch
from PIL import Image

import sys
import os
import torch
# import sys
# sys.path.append('t3d')
# from launch import *

# Todo: Integrate threestudio into this file
text2img3d_model = {
    "ldm3d"                 : ("LDM3D", "Intel/ldm3d"),
    "ldm3d_4c"              : ("LDM3D_4C","Intel/ldm3d-4c"),
    # Waiting for the bug to be PR.
    # "ldm3d_sr"              : ("LDM3D_SR", [ "Intel/ldm3d-4c","Intel/ldm3d-sr"]),
    "ldm3d_pano"            : ("LDM3D_Pano", "Intel/ldm3d-pano"),
    "ShapE"                 : ("ShapE", "openai/shap-e"),
    "dreamfusion-sd"        : ("Dreamfusion_sd", "t3d/configs/dreamfusion-sd.yaml") ,
    "dreamfusion-if"        : ("Dreamfusion_if", "t3d/configs/dreamfusion-if.yaml"),
    "prolificdreamer"       : ("Prolificdreamer", ["t3d/configs/prolificdreamer.yaml", "t3d/configs/prolificdreamer-geometry.yaml", "t3d/configs/prolificdreamer-texture.yaml"]),
    "magic3d-if"            : ("Magic3D", ["t3d/configs/magic3d-coarse-if.yaml", "t3d/configs/magic3d-refine-sd.yaml"]),
    "magic3d-sd"            : ("Magic3D", ["t3d/configs/magic3d-coarse-sd.yaml", "t3d/configs/magic3d-refine-sd.yaml"]),
    "sjc"                   : ("SJC", "t3d/configs/sjc.yaml"),
    "latentnerf"            : ("LatentNeRF", ["t3d/configs/latentnerf.yaml", "t3d/configs/latentnerf-refine.yaml"]),
    "fantasia3d"            : ("Fantasia3D", ["t3d/configs/fantasia3d.yaml", "t3d/configs/fantasia3d-texture.yaml"]),
    "textmesh"              : ("TextMesh", ["t3d/configs/textmesh-if.yaml"]),
    "dreamfusion-hifa"      : ("DreamFusion_HiFA", "t3d/configs/hifa.yaml"),
    "prolificdreamer-hifa"  : ("ProlificDreamer_HiFA", "t3d/configs/prolificdreamer-hifa.yaml")    ,
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
    def text_to_3D(self, prompt):
        imgs = self.model.text_to_3D(prompt)
        return imgs
    

class LDM3D(AbstractModel):
    def __init__(self, ckpt: str = 'Intel/ldm3d', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionLDM3DPipeline
        self.pipeline = StableDiffusionLDM3DPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
    def text_to_3D(self, prompt):
        output = self.pipeline(prompt)
        return output.rgb[0], output.depth[0]


class LDM3D_4C(AbstractModel):
    def __init__(self, ckpt: str = "Intel/ldm3d-4c", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionLDM3DPipeline
        self.pipeline = StableDiffusionLDM3DPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
    def text_to_3D(self, prompt):
        output = self.pipeline(prompt)
        output_rgb, output_depth = output.rgb, output.depth
        return output_rgb[0], output_depth[0]
    
class LDM3D_Pano(AbstractModel):
    def __init__(self, ckpt: str = "Intel/ldm3d-pano", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionLDM3DPipeline
        self.pipeline = StableDiffusionLDM3DPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
    def text_to_3D(self, prompt):
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
    def text_to_3D(self, prompt):
        output = self.pipeline_ldm3d(prompt)
        output_rgb, output_depth = output.rgb[0].convert("RGB"), output.depth[0].convert("L")
        outputs = self.pipeline_upscale(prompt="high quality high resolution uhd 4k image", rgb=output_rgb, depth=output_depth, num_inference_steps=50, target_res=[1024, 1024])
        output_rgb, output_depth = outputs.rgb[0], outputs.depth[0]
        return output_rgb, output_depth
    
    
    
class ShapE(AbstractModel):
    def __init__(self, ckpt: str = "openai/shap-e", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import DiffusionPipeline
        self.pipeline = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
    def text_to_3D(self, prompt):
        images = self.pipeline(prompt, guidance_scale=15.0, num_inference_steps=64, frame_size=256).images
        return images[0]
    
    
class Dreamfusion_sd(AbstractModel):
    def __init__(self, ckpt: str = "t3d/configs/dreamfusion-sd.yaml", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def text_to_3D(self, prompt):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f
        input_ns.gpu = self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        result = main(input_ns, extras)
        return result[0]
    
class Dreamfusion_if(AbstractModel):
    def __init__(self, ckpt: str = "t3d/configs/dreamfusion-if.yaml", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def text_to_3D(self, prompt):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f
        input_ns.gpu = self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        result = main(input_ns, extras)
        return result[0]

    

class Prolificdreamer(AbstractModel):
    def __init__(self, ckpt: str = "t3d/configs/prolificdreamer.yaml", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def text_to_3D(self, prompt, geometry_refine = True, texturing = False,**kwargs):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        # pass kwargs to extras
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
            
        result = main(input_ns, extras)

        return result[0]
    
    

class Magic3D(AbstractModel):
    def __init__(self, ckpt: str = ["t3d/configs/magic3d-coarse-if.yaml", "t3d/configs/magic3d-refine-sd.yaml"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def text_to_3D(self, prompt, refine = True, **kwargs):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f[0]
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        # pass kwargs to extras
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
            
        result = main(input_ns, extras)
        if refine:
            input_ns.config = self.config_f[1]
            extras.append(f"system.geometry_convert_from={result[2]}")
            # system.geometry_convert_override.isosurface_threshold=some_value 0-20.
            # extras.append(f"system.geometry_convert_override.isosurface_threshold={20.0}")
            result = main(input_ns, extras)
        return result[0]
    
class SJC(AbstractModel):
    def __init__(self, ckpt: str = "t3d/configs/sjc.yaml", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def text_to_3D(self, prompt, refine = True, **kwargs):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        # pass kwargs to extras
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
            
        result = main(input_ns, extras)
        return result[0]
    
class LatentNeRF(AbstractModel):
    def __init__(self, ckpt: list = ["t3d/configs/latentnerf.yaml", "t3d/configs/latentnerf-refine.yaml"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def text_to_3D(self, prompt, refine = True, **kwargs):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f[0]
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        # pass kwargs to extras
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
            
        result = main(input_ns, extras)
        if refine:
            input_ns.config = self.config_f[1]
            extras.append(f"system.weights={result[2]}")
            # system.geometry_convert_override.isosurface_threshold=some_value 0-20.
            # extras.append(f"system.geometry_convert_override.isosurface_threshold={20.0}")
            result = main(input_ns, extras)
        return result[0]
    
class Fantasia3D(AbstractModel):
    def __init__(self, ckpt: list = ["t3d/configs/fantasia3d.yaml", "t3d/configs/fantasia3d-texture.yaml"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def text_to_3D(self, prompt, texture = True, **kwargs):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f[0]
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        # pass kwargs to extras
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
            
        result = main(input_ns, extras)
        if texture:
            input_ns.config = self.config_f[1]
            extras.append(f"system.geometry_convert_from={result[2]}")
            result = main(input_ns, extras)
        return result[0]
    
class TextMesh(AbstractModel):
    def __init__(self, ckpt: list = ["t3d/configs/textmesh-if.yaml"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def text_to_3D(self, prompt, **kwargs):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f[0]
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        # pass kwargs to extras
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
            
        result = main(input_ns, extras)
        return result[0]
    
class DreamFusion_HiFA(AbstractModel):
    def __init__(self, ckpt: str  = "t3d/configs/hifa.yaml", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def text_to_3D(self, prompt, **kwargs):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f[0]
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        # pass kwargs to extras
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
            
        result = main(input_ns, extras)
        
        return result[0]

class ProlificDreamer_HiFA(AbstractModel):
    def __init__(self, ckpt: str  = "t3d/configs/prolificdreamer-hifa.yaml", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def text_to_3D(self, prompt, **kwargs):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f[0]
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        # pass kwargs to extras
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
            
        result = main(input_ns, extras)
        
        return result[0]