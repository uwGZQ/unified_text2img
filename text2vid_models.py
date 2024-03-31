import base64
import json
import os
import random
from typing import Union

import numpy as np
import torch
from PIL import Image

text2vid_model = {
    "text2vid-zero":                ("Text2VideoZero", "runwayml/stable-diffusion-v1-5"),
    "zeroscope":                    ("ZeroScope", ["cerspense/zeroscope_v2_576w", "cerspense/zeroscope_v2_XL"]),
    "modelscope-t2v":               ("ModelScopeT2V", "damo-vilab/text-to-video-ms-1.7b"),
    "animatediff":                  ("AnimateDiff", ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE"]),
    "animatediff-motion-lora":      ("AnimateDif_motion_lora", ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE", "guoyww/animatediff-motion-lora-zoom-out", "zoom-out"]),
    "animatediff-motion-lora-peft": ("AnimateDif_motion_lora_peft", ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE", "guoyww/animatediff-motion-lora-zoom-out", "zoom-out", "guoyww/animatediff-motion-lora-pan-left", "pan-left"]),
    "animateLCM":                   ("AnimateLCM", ["wangfuyun/AnimateLCM", "emilianJR/epiCRealism", "AnimateLCM_sd15_t2v_lora.safetensors", "lcm-lora"]),
    "animateLCM-motion-lora":       ("AnimateLCM_motion_lora", ["wangfuyun/AnimateLCM", "emilianJR/epiCRealism", "AnimateLCM_sd15_t2v_lora.safetensors", "lcm-lora", "guoyww/animatediff-motion-lora-tilt-up", "tilt-up"]),
    "free-init":                    ("FreeInit", ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE"]),
}



def set_model_key(model_name, key):
	text2vid_model[model_name] = (text2vid_model[model_name][0], key)


def list_text2vid_models():
	return list(text2vid_model.keys())

class AbstractModel:
    def text2vid(self, prompt):
        "(Abstract method) abstract text2vid method"

class Text2Vid:
    def __init__(self, model_name: str = "stable-diffusion-2-1", model: AbstractModel = None, ckpt: str="stabilityai/stable-diffusion-2-1" , precision: torch.dtype = torch.float16, torch_device: str = "cuda", calculate_metrics: bool = False):
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
            class_name, ckpt = text2vid_model[model_name]
            self.model_presision = precision
            self.model = eval(class_name)(ckpt, precision, torch_device)
            print(f"Finish loading {model_name}")
        else:
            print(f"Using provided model ...")
            class_name = model.__class__.__name__
            self.model = eval(class_name)(ckpt, precision, torch_device)

    @torch.no_grad()
    def text2vid(self, prompt):
        vid = self.model.text2vid(prompt)
        # vid = [(r * 255).astype("uint8") for r in vid]
        return vid

    @torch.no_grad()
    def save_vid(self, video_frames, path_str, fps=4):
        if isinstance(video_frames[0] ,np.ndarray):

            result = [(r * 255).astype("uint8") for r in video_frames]
        elif isinstance(video_frames[0] ,torch.Tensor):
            result = [(r.cpu().numpy() * 255).astype("uint8") for r in video_frames]


        import imageio
        state = imageio.mimsave(path_str, video_frames, fps=fps)
        return state





class Text2VideoZero(AbstractModel):
    def __init__(self, ckpt:str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import TextToVideoZeroPipeline
        self.pipeline = TextToVideoZeroPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
    def text2vid(self, prompt):
        result = self.pipeline(prompt).images
        return result

class ZeroScope(AbstractModel):
    def __init__(self, ckpt: list = ["cerspense/zeroscope_v2_576w","cerspense/zeroscope_v2_XL"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        from diffusers.utils import export_to_video

        self.pipeline = DiffusionPipeline.from_pretrained(ckpt[0], torch_dtype=precision).to(device)
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
        self.pipeline.enable_vae_slicing() 

        self.upscale = DiffusionPipeline.from_pretrained(ckpt[1], torch_dtype=torch.float16).to(device)
        self.upscale.scheduler = DPMSolverMultistepScheduler.from_config(self.upscale.scheduler.config)
        self.upscale.enable_model_cpu_offload()
        self.upscale.unet.enable_forward_chunking(chunk_size=1, dim=1)
        self.upscale.enable_vae_slicing()

    def text2vid(self, prompt):
        video_frames = self.pipeline(prompt, num_frames=24).frames[0]
        video = [Image.fromarray((frame*255).astype(np.uint8)).resize((1024, 576)) for frame in video_frames]
        video_frames = self.upscale(prompt, video=video, strength=0.6).frames[0]

        return video_frames

class ModelScopeT2V(AbstractModel):
    def __init__(self, ckpt: str = "damo-vilab/text-to-video-ms-1.7b", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"),EnableVAESlicing=True):
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video 
        self.pipeline = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
        self.pipeline.enable_model_cpu_offload()
        if EnableVAESlicing:
            self.pipeline.enable_vae_slicing()


    def text2vid(self, prompt):
        result = self.pipeline(prompt, num_frames = 64).frames[0]
        return result

class AnimateDiff(AbstractModel):
    def __init__(self, ckpt: list = ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        model_id = ckpt[1]
        self.pipeline = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=precision).to(device)
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1
        )
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.enable_model_cpu_offload()
    def text2vid(self, prompt):
        output = self.pipeline(prompt, num_frames=16, guidance_scale=7.5, num_inference_steps=25, generator=torch.Generator("cpu").manual_seed(42))
        result = output.frames[0]
        return result

class AnimateDif_motion_lora(AbstractModel):
    def __init__(self, ckpt: list = ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE", "guoyww/animatediff-motion-lora-zoom-out", "zoom-out"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        model_id = ckpt[1]
        self.pipeline = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=precision).to(device)
        self.pipeline.load_lora_weights(ckpt[2], adapter_name=ckpt[3])
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1
        )
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()
    def text2vid(self, prompt):
        output = self.pipeline(prompt, num_frames=16, guidance_scale=7.5, num_inference_steps=25, generator=torch.Generator("cpu").manual_seed(42))
        result = output.frames[0]
        return result

class AnimateDif_motion_lora_peft(AbstractModel):
    def __init__(self, ckpt: list = ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE", "guoyww/animatediff-motion-lora-zoom-out", "zoom-out", "guoyww/animatediff-motion-lora-pan-left", "pan-left"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        model_id = ckpt[1]
        self.pipeline = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=precision).to(device)
        self.pipeline.load_lora_weights(ckpt[2], adapter_name=ckpt[3])
        self.pipeline.load_lora_weights(ckpt[4], adapter_name=ckpt[5])
        self.pipeline.set_adapters([ckpt[3], ckpt[5]], adapter_weights=[1.0, 1.0])
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1
        )
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()
    def text2vid(self, prompt):
        output = self.pipeline(prompt, num_frames=16, guidance_scale=7.5, num_inference_steps=25, generator=torch.Generator("cpu").manual_seed(42))
        result = output.frames[0]
        return result

class AnimateLCM(AbstractModel):
    def __init__(self, ckpt: list = ["wangfuyun/AnimateLCM", "emilianJR/epiCRealism", "AnimateLCM_sd15_t2v_lora.safetensors", "lcm-lora"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        self.pipeline = AnimateDiffPipeline.from_pretrained(ckpt[1], motion_adapter=adapter, torch_dtype=precision).to(device)
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config, beta_schedule="linear")
        self.pipeline.load_lora_weights(ckpt[0], weight_name=ckpt[2], adapter_name=ckpt[3])
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()

    def text2vid(self, prompt):
        output = self.pipeline(prompt, num_frames=16, guidance_scale=1.5, num_inference_steps=6, generator=torch.Generator("cpu").manual_seed(0))
        result = output.frames[0]
        return result

class AnimateLCM_motion_lora(AbstractModel):
    def __init__(self, ckpt: list = ["wangfuyun/AnimateLCM", "emilianJR/epiCRealism", "AnimateLCM_sd15_t2v_lora.safetensors", "lcm-lora", "guoyww/animatediff-motion-lora-tilt-up", "tilt-up"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        self.pipeline = AnimateDiffPipeline.from_pretrained(ckpt[1], motion_adapter=adapter, torch_dtype=precision).to(device)
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config, beta_schedule="linear")
        self.pipeline.load_lora_weights(ckpt[0], weight_name=ckpt[2], adapter_name=ckpt[3])
        self.pipeline.load_lora_weights(ckpt[4], adapter_name=ckpt[5])
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload() 
    def text2vid(self, prompt):
        output = self.pipeline(prompt, num_frames=16, guidance_scale=1.5, num_inference_steps=6, generator=torch.Generator("cpu").manual_seed(0))
        result = output.frames[0]
        return result

class FreeInit(AbstractModel):
    def __init__(self, ckpt: list = ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
        from diffusers.utils import export_to_gif
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        model_id = ckpt[1]
        self.pipeline = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=precision).to(device)
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1
        )
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.enable_free_init(method="butterworth", use_fast_sampling=True)
    def text2vid(self, prompt):
        output = self.pipeline(prompt, num_frames=16, guidance_scale=7.5, num_inference_steps=20, generator=torch.Generator("cpu").manual_seed(666))
        self.pipeline.disable_free_init()

        result = output.frames[0]

        return result

