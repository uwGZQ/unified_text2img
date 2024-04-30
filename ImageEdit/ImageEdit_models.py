import base64
import json
import os
import random
from typing import Union

import numpy as np
import torch
from PIL import Image


from torch import autocast, inference_mode

imageedit_model = {
    "instructpix2pix":                  ("InstructPix2Pix", "timbrooks/instruct-pix2pix"),
    "ledits-pp-sd":                     ("LEditsPP_sd", "runwayml/stable-diffusion-v1-5"),
    "ledits-pp-xl":                     ("LEditsPP_sdxl", ["stabilityai/stable-diffusion-xl-base-1.0", "madebyollin/sdxl-vae-fp16-fix"]),
    "ledits":                           ("LEDITS", "runwayml/stable-diffusion-v1-5"),
    "pix2pix-zero":                     ("Pix2PixZero", ["Salesforce/blip-image-captioning-base", "CompVis/stable-diffusion-v1-4"]),
    "ddim":                             ("DDIM", "CompVis/stable-diffusion-v1-4"),
    "prompt_to_prompt_ddim":            ("PromptToPromptDDIM", "CompVis/stable-diffusion-v1-4"),
    "prompt_to_prompt_inversion":       ("PromptToPromptInversion", "CompVis/stable-diffusion-v1-4"),
    "ddpm_inversion":                   ("DDPMInversion", "CompVis/stable-diffusion-v1-4"),
    "score_distillation_sampling":      ("ScoreDistillationSampling", "runwayml/stable-diffusion-v1-5"),
    "delta_denoising_score_zero_shot":  ("DDS_zero_shot", "runwayml/stable-diffusion-v1-5"),
}




def set_model_key(model_name, key):
	imageedit_model[model_name] = (imageedit_model[model_name][0], key)


def list_imageedit_models():
	return list(imageedit_model.keys())

class AbstractModel:
    def imageedit(self, prompt):
        "(Abstract method) abstract image_edit method"

class ImageEdit:
    def __init__(self, model_name: str = "instructpix2pix", model: AbstractModel = None, ckpt: str="timbrooks/instruct-pix2pix" , precision: torch.dtype = torch.float16, torch_device: str = "cuda", calculate_metrics: bool = False):
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
            class_name, ckpt = imageedit_model[model_name]
            self.model_presision = precision
            self.model = eval(class_name)(ckpt, precision, torch_device)
            print(f"Finish loading {model_name}")
        else:
            print(f"Using provided model ...")
            class_name = model.__class__.__name__
            self.model = eval(class_name)(ckpt, precision, torch_device)

    # @torch.no_grad()
    def editimage(self, prompt, image, **kwargs):
        from diffusers.utils import load_image
        image = load_image(image).resize((512, 512))
        result = self.model.editimage(prompt=prompt, image = image, **kwargs)
        # vid = [(r * 255).astype("uint8") for r in vid]
        return result


class InstructPix2Pix(AbstractModel):
    def __init__(self, ckpt:str = "timbrooks/instruct-pix2pix", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionInstructPix2PixPipeline
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt, torch_dtype=torch.float16).to(device)
    def editimage(self, prompt, image, **kwargs):
        result = self.pipeline(prompt=prompt, image=image, **kwargs).images
        return result[0]
    

class LEditsPP_sd(AbstractModel):
    def __init__(self, ckpt:str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import LEditsPPPipelineStableDiffusion
        self.pipeline = LEditsPPPipelineStableDiffusion.from_pretrained(ckpt, torch_dtype=torch.float16).to(device)
    def editimage(self, prompt, image, **kwargs):
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        _ = self.pipeline.invert(image=image, num_inversion_steps=50, skip=0.1)
        #edit_guidance_scale=10.0, edit_threshold=0.75).images[0]
        result = self.pipeline(editing_prompt=prompt, **kwargs).images
        return result[0]

class LEditsPP_sdxl(AbstractModel):
    def __init__(self, ckpt:list = ["stabilityai/stable-diffusion-xl-base-1.0", "madebyollin/sdxl-vae-fp16-fix"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import LEditsPPPipelineStableDiffusionXL
        if precision == torch.float16:
            from diffusers import DDIMScheduler, AutoencoderKL
            print(ckpt)
            vae = AutoencoderKL.from_pretrained(ckpt[1], torch_dtype=torch.float16)
            scheduler = DDIMScheduler.from_pretrained(ckpt[0], subfolder="scheduler")
            self.pipeline = LEditsPPPipelineStableDiffusionXL.from_pretrained(ckpt[0],vae = vae, scheduler=scheduler, torch_dtype = precision, variant="fp16").to(device)
        elif precision == torch.float32:
            self.pipeline = LEditsPPPipelineStableDiffusionXL.from_pretrained(ckpt[0], torch_dtype=precision).to(device)
    def editimage(self, prompt, image, **kwargs):
        # max_edge = max(image.size)
        # image = image.resize((max_edge, max_edge))
        _ = self.pipeline.invert(image=image, num_inversion_steps=50, skip=0.2)
        #reverse_editing_direction=[True,False],edit_guidance_scale=[5.0,10.0],edit_threshold=[0.9,0.85],
        result = self.pipeline(editing_prompt=prompt, **kwargs).images
        return result[0]
    
    
class LEDITS(AbstractModel):
    # Currently only support torch.float32
    def __init__(self, ckpt:str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float32, device: torch.device = torch.device("cuda")):
        assert precision == torch.float32
        self.device = device
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        from LEDITS_Utils import SemanticStableDiffusionPipeline
        
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
        self.sd_pipe.scheduler = DDIMScheduler.from_config(ckpt, subfolder="scheduler")
        self.sega_pipe = SemanticStableDiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
    def invert(self,  x0:torch.FloatTensor, num_diffusion_steps=100, prompt_src:str ="", num_inference_steps=100, cfg_scale_src = 3.5, eta = 1):
        from LEDITS_Utils import inversion_forward_process

        from torch import autocast, inference_mode
          
        self.sd_pipe.scheduler.set_timesteps(num_diffusion_steps)
        with autocast("cuda"), inference_mode():
            w0 = (self.sd_pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()
        wt, zs, wts = inversion_forward_process(self.sd_pipe, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=num_diffusion_steps)
        return zs, wts
    def sample(self,zs, wts, prompt_tar="", cfg_scale_tar=15, skip=36, eta = 1):
        from LEDITS_Utils import inversion_reverse_process, image_grid
        from torch import autocast, inference_mode
        w0, _ = inversion_reverse_process(self.sd_pipe, xT=wts[skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[skip:])
        with autocast("cuda"), inference_mode():
            x0_dec = self.sd_pipe.vae.decode(1 / 0.18215 * w0).sample
        if x0_dec.dim()<4:
            x0_dec = x0_dec[None,:,:,:]
        img = image_grid(x0_dec)
        return img
    def edit(self, wts, zs,
            tar_prompt = "",
            steps = 100,
            skip = 36,
            tar_cfg_scale =15,
            edit_concept = "",
            guidnace_scale = 7,
            warmup = 1,
            neg_guidance=False,
            threshold=0.95

   ):
        editing_args = dict(
            editing_prompt = edit_concept,
            reverse_editing_direction = neg_guidance,
            edit_warmup_steps=warmup,
            edit_guidance_scale=guidnace_scale,
            edit_threshold=threshold,
            edit_momentum_scale=0.5,
            edit_mom_beta=0.6,
            eta=1,
        )
        latnets = wts[skip].expand(1, -1, -1, -1)
        sega_out = self.sega_pipe(prompt=tar_prompt, latents=latnets, guidance_scale = tar_cfg_scale,
                            num_images_per_prompt=1,
                            num_inference_steps=steps,
                            use_ddpm=True,  wts=wts, zs=zs[skip:], **editing_args)
        print(tar_prompt)
        return sega_out.images[0]
    def editimage(self, prompt, image, num_diffusion_steps =100,source_guidance_scale = 3.5, reconstruct = True, skip_steps =36, target_guidance_scale=20, edit_guidance_scales=[7,15], warmup_steps=[1,1], reverse_editing=[True, False], thresholds = [ 0.95,0.95]):
        from LEDITS_Utils import load_512
        seed = 36478574352
        x0 = load_512(image, device=self.device)
        source_prompt = ""
        target_prompt = ""
        edit_concepts = []
        if isinstance(prompt, str):
            edit_concepts = [prompt] * len(edit_guidance_scales)
        elif isinstance(prompt, dict):
            for key in prompt:
                if "source" in key:
                    source_prompt = prompt[key]
                elif "target" in key:
                    target_prompt = prompt[key]
                elif "edit" in key:
                    edit_concepts=prompt[key]
                
        assert len(edit_guidance_scales) == len(warmup_steps) == len(reverse_editing) == len(thresholds) == len(edit_concepts)

        zs, wts = self.invert(x0 =x0 , num_diffusion_steps=100, prompt_src=source_prompt, num_inference_steps=num_diffusion_steps, cfg_scale_src=source_guidance_scale)
        print(target_prompt)
        if reconstruct:
            ddpm_out_img = self.sample(zs, wts, prompt_tar=target_prompt, skip=skip_steps, cfg_scale_tar=target_guidance_scale)
        
        sega_ddpm_edited_img = self.edit(wts, zs, tar_prompt=target_prompt, steps=num_diffusion_steps, skip=skip_steps, tar_cfg_scale=target_guidance_scale, edit_concept=edit_concepts, guidnace_scale=edit_guidance_scales, warmup=warmup_steps, neg_guidance=reverse_editing, threshold=thresholds)
        return sega_ddpm_edited_img
    

class Pix2PixZero(AbstractModel):
    def __init__(self, ckpt:list = ["Salesforce/blip-image-captioning-base", "CompVis/stable-diffusion-v1-4"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from transformers import BlipForConditionalGeneration, BlipProcessor
        from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline
        self.processor = BlipProcessor.from_pretrained(ckpt[0])
        self.model = BlipForConditionalGeneration.from_pretrained(ckpt[0], torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(ckpt[1], caption_generator=self.model, caption_processor=self.processor, torch_dtype=torch.float16, safety_checker=None).to(device)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.enable_model_cpu_offload()
        
    def editimage(self, prompt, image, **kwargs):
        if isinstance(prompt, str):
            raise ValueError("Prompt should be a dictionary with 'source_prompt' and 'target_prompt' keys")
        
        source_prompts = prompt["source_prompt"]
        target_prompts = prompt["target_prompt"]
        caption = self.pipeline.generate_caption(image)
        inv_latents = self.pipeline.invert(caption, image=image).latents
        source_embeds = self.pipeline.get_embeds(source_prompts)
        target_embeds = self.pipeline.get_embeds(target_prompts)
        # num_inference_steps=50, cross_attention_guidance_amount=0.15, 
        result = self.pipeline(caption, source_embeds=source_embeds, target_embeds=target_embeds, latents=inv_latents, negative_prompt=caption, num_inference_steps=50, cross_attention_guidance_amount=0.15, ).images[0]
        return result
    
class PromptToPromptDDIM(AbstractModel):
    def __init__(self, ckpt:str = "CompVis/stable-diffusion-v1-4", precision: torch.dtype = torch.float32, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        cfg_scale_src = 3.5
        cfg_scale_tar_list = [15]
        eta = 1
        skip_zs = [0]
        xa_sa_string = '_'
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(ckpt, precision=precision).to(device)
        self.device = device
    def editimage(self, prompt, image, num_diffusion_steps=100, **kwargs):
        if isinstance(prompt, str):
            prompt = {"source_prompt": "", "target_prompt": prompt}
        prompt_src = prompt.get("source_prompt","")
        prompt_tar = prompt["target_prompt"] # hope to be a string
        cfg_scale_src = 3.5
        cfg_scale_tar_list = [15]
        eta = 1
        skip_zs = [0]
        xa_sa_string = '_'
        from diffusers import DDIMScheduler
        from prompt_to_prompt.ptp_classes import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl,load_512
        from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable, view_images
        from ddm_inversion.inversion_utils import  inversion_forward_process, inversion_reverse_process
        from ddm_inversion.utils import image_grid,dataset_from_yaml,tensor_to_pil
        from ddm_inversion.ddim_inversion import ddim_inversion
                

        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.ldm_stable.scheduler = scheduler
        self.ldm_stable.scheduler.set_timesteps(num_diffusion_steps)
        
        offsets=(0,0,0,0)
        x0 = load_512(image, *offsets, self.device)
        
        with autocast("cuda"), inference_mode():
            w0 = (self.ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()
            
        wT = ddim_inversion(self.ldm_stable, w0, prompt_src, cfg_scale_src)

        src_tar_len_eq = (len(prompt_src.split(" ")) == len(prompt_tar.split(" ")))
        print(cfg_scale_tar_list,skip_zs)
        for cfg_scale_tar in cfg_scale_tar_list:
            for skip in skip_zs:
                if skip != 0:
                    print(skip)
                    continue
                    print("here")
                prompts = [prompt_src, prompt_tar]
                if src_tar_len_eq:
                    controller = AttentionReplace(prompts, num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=self.ldm_stable)
                else:
                    controller = AttentionRefine(prompts, num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=self.ldm_stable)


                register_attention_control(self.ldm_stable, controller)
                cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                w0, latent = text2image_ldm_stable(self.ldm_stable, prompts, controller, num_diffusion_steps, cfg_scale_list, None, wT)
                w0 = w0[1:2]
                
                with autocast("cuda"), inference_mode():
                    x0_dec = self.ldm_stable.vae.decode(1 / 0.18215 * w0).sample
                    return tensor_to_pil(x0_dec)[0]

class DDIM(AbstractModel):
    def __init__(self, ckpt:str = "CompVis/stable-diffusion-v1-4", precision: torch.dtype = torch.float32, device: torch.device = torch.device("cuda"), num_diffusion_steps=100,):
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        cfg_scale_src = 3.5
        cfg_scale_tar_list = [15]
        eta = 1
        skip_zs = [0]
        xa_sa_string = '_'
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(ckpt, precision=precision).to(device)
        self.num_diffusion_steps = num_diffusion_steps

        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.ldm_stable.scheduler = scheduler
        self.ldm_stable.scheduler.set_timesteps(num_diffusion_steps)
        
        self.device = device
    def editimage(self, prompt, image, num_diffusion_steps=100, **kwargs):
        if isinstance(prompt, str):
            prompt = {"source_prompt": "", "target_prompt": prompt}
        prompt_src = prompt.get("source_prompt","")
        prompt_tar = prompt["target_prompt"] # hope to be a string
        cfg_scale_src = 3.5
        cfg_scale_tar_list = [15]
        eta = 1
        skip_zs = [0]
        xa_sa_string = '_'
        from diffusers import DDIMScheduler
        from prompt_to_prompt.ptp_classes import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl,load_512
        from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable, view_images
        from ddm_inversion.inversion_utils import  inversion_forward_process, inversion_reverse_process
        from ddm_inversion.utils import image_grid,dataset_from_yaml, tensor_to_pil
        from ddm_inversion.ddim_inversion import ddim_inversion
        
        offsets=(0,0,0,0)
        x0 = load_512(image, *offsets, self.device)
        
        with autocast("cuda"), inference_mode():
            w0 = (self.ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()
            
        wT = ddim_inversion(self.ldm_stable, w0, prompt_src, cfg_scale_src)

        src_tar_len_eq = (len(prompt_src.split(" ")) == len(prompt_tar.split(" ")))

        for cfg_scale_tar in cfg_scale_tar_list:
            for skip in skip_zs:
                if skip != 0:
                    continue
                prompts = [prompt_src, prompt_tar]
                controller = EmptyControl()
                register_attention_control(self.ldm_stable, controller)
                cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                w0, latent = text2image_ldm_stable(self.ldm_stable, prompts, controller,self.num_diffusion_steps, cfg_scale_list, None, wT)
                w0 = w0[1:2]
                    
                with autocast("cuda"), inference_mode():
                    x0_dec = self.ldm_stable.vae.decode(1 / 0.18215 * w0).sample
                    return tensor_to_pil(x0_dec)[0]
              
class PromptToPromptInversion(AbstractModel):
    def __init__(self, ckpt:str = "CompVis/stable-diffusion-v1-4", precision: torch.dtype = torch.float32,  device: torch.device = torch.device("cuda"), num_diffusion_steps = 100,):
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        cfg_scale_src = 3.5
        cfg_scale_tar_list = [15]
        eta = 1
        skip_zs = [36]
        xa_sa_string = '_'
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(ckpt, precision=precision).to(device)
        
        self.ldm_stable.scheduler = DDIMScheduler.from_config(ckpt, subfolder = "scheduler")
        
        self.ldm_stable.scheduler.set_timesteps(num_diffusion_steps)
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
    def editimage(self, prompt, image, num_diffusion_steps=100, **kwargs):
        if isinstance(prompt, str):
            raise ValueError("Prompt should be a dictionary with 'source_prompt' and 'target_prompt' keys")
        prompt_src = prompt.get("source_prompt","")
        prompt_tar = prompt["target_prompt"] # hope to be a string
        cfg_scale_src = 3.5
        cfg_scale_tar_list = [15]
        eta = 1
        skip_zs = [36]
        xa = 0.6
        sa = 0.2
        from diffusers import DDIMScheduler
        from prompt_to_prompt.ptp_classes import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl,load_512
        from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable, view_images
        from ddm_inversion.inversion_utils import  inversion_forward_process, inversion_reverse_process
        from ddm_inversion.utils import image_grid, dataset_from_yaml, tensor_to_pil
        from ddm_inversion.ddim_inversion import ddim_inversion
        
        offsets=(0,0,0,0)
        x0 = load_512(image, *offsets, self.device)
        
        with autocast("cuda"), inference_mode():
            w0 = (self.ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()
            
        wt, zs, wts = inversion_forward_process(self.ldm_stable, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=self.num_diffusion_steps)

        src_tar_len_eq = (len(prompt_src.split(" ")) == len(prompt_tar.split(" ")))
        
        

        for cfg_scale_tar in cfg_scale_tar_list:
            for skip in skip_zs:
                cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                prompts = [prompt_src, prompt_tar]
                if src_tar_len_eq:
                    controller = AttentionReplace(prompts, self.num_diffusion_steps, cross_replace_steps=xa, self_replace_steps=sa, model=self.ldm_stable)
                else:
                    # Should use Refine for target prompts with different number of tokens
                    controller = AttentionRefine(prompts, self.num_diffusion_steps, cross_replace_steps=xa, self_replace_steps=sa, model=self.ldm_stable)
                    
                register_attention_control(self.ldm_stable, controller)
                w0, _ = inversion_reverse_process(self.ldm_stable, xT=wts[self.num_diffusion_steps-skip], etas=eta, prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[:(self.num_diffusion_steps-skip)], controller=controller)
                w0 = w0[1].unsqueeze(0)
                    
                with autocast("cuda"), inference_mode():
                    x0_dec = self.ldm_stable.vae.decode(1 / 0.18215 * w0).sample
                    return tensor_to_pil(x0_dec)[0]
                
                
class DDPMInversion(AbstractModel):
    def __init__(self, ckpt:str = "CompVis/stable-diffusion-v1-4", precision: torch.dtype = torch.float32, device: torch.device = torch.device("cuda"),num_diffusion_steps = 100,):
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        cfg_scale_src = 3.5
        cfg_scale_tar_list = [15]
        eta = 1
        skip_zs = [36]
        xa_sa_string = '_'
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(ckpt, precision=precision).to(device)
        
        self.ldm_stable.scheduler = DDIMScheduler.from_config(ckpt, subfolder = "scheduler")
        
        self.ldm_stable.scheduler.set_timesteps(num_diffusion_steps)
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
    def editimage(self, prompt, image, num_diffusion_steps=100, **kwargs):
        if isinstance(prompt, str):
            raise ValueError("Prompt should be a dictionary with 'source_prompt' and 'target_prompt' keys")
        prompt_src = prompt.get("source_prompt","")
        prompt_tar = prompt["target_prompt"] # hope to be a string
        cfg_scale_src = 3.5
        cfg_scale_tar_list = [15]
        eta = 1
        skip_zs = [36]
        xa = 0.6
        sa = 0.2
        from diffusers import DDIMScheduler
        from prompt_to_prompt.ptp_classes import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl,load_512
        from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable, view_images
        from ddm_inversion.inversion_utils import  inversion_forward_process, inversion_reverse_process
        from ddm_inversion.utils import image_grid, dataset_from_yaml, tensor_to_pil
        from ddm_inversion.ddim_inversion import ddim_inversion

        offsets=(0,0,0,0)
        x0 = load_512(image, *offsets, self.device)
        
        with autocast("cuda"), inference_mode():
            w0 = (self.ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()
            
        wt, zs, wts = inversion_forward_process(self.ldm_stable, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=self.num_diffusion_steps)

        src_tar_len_eq = (len(prompt_src.split(" ")) == len(prompt_tar.split(" ")))
        
        

        for cfg_scale_tar in cfg_scale_tar_list:
            for skip in skip_zs:
                controller = AttentionStore()
                register_attention_control(self.ldm_stable, controller)
                w0, _ = inversion_reverse_process(self.ldm_stable, xT=wts[self.num_diffusion_steps-skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[:(self.num_diffusion_steps-skip)], controller=controller)

                with autocast("cuda"), inference_mode():
                    x0_dec = self.ldm_stable.vae.decode(1 / 0.18215 * w0).sample
                    return tensor_to_pil(x0_dec)[0]

            
class ScoreDistillationSampling(AbstractModel):
    def __init__(self, ckpt: str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionPipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(ckpt, precision=precision).to(device)
        self.device = device
    def editimage(self, prompt, image, num_iters=200):
        if isinstance(image, Image.Image):
            # img to numpy array
            image = np.array(image)
        text_source = prompt.get("source_prompt","")
        text_target = prompt["target_prompt"]
        
        from DDS_zeroshot_utils import get_text_embeddings, DDSLoss, decode
        from torch.optim.adamw import AdamW
        from torch.optim.sgd import SGD
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel
        dds_loss = DDSLoss(self.device, self.pipeline)
        image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
        image_source = image_source.unsqueeze(0).to(self.device)
        with torch.no_grad():
            z_source = self.pipeline.vae.encode(image_source)['latent_dist'].mean * 0.18215
            image_target = image_source.clone()
            embedding_null = get_text_embeddings(self.pipeline, "")
            embedding_text = get_text_embeddings(self.pipeline, text_source)
            embedding_text_target = get_text_embeddings(self.pipeline, text_target)
            embedding_source = torch.stack([embedding_null, embedding_text], dim=1)
            embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)
        guidance_scale = 7.5
        image_target.requires_grad = True
        use_dds_loss = True

        z_taregt = z_source.clone()
        z_taregt.requires_grad = True
        optimizer = SGD(params=[z_taregt], lr=1e-1)
        # use_dds = False
        for i in range(num_iters):
            loss, log_loss = dds_loss.get_sds_loss(z_taregt, embedding_target)
            optimizer.zero_grad()
            (2000 * loss).backward()
            optimizer.step()
            
        out = decode(z_taregt, self.pipeline, im_cat=image)
        return out





class DDS_zero_shot(AbstractModel):
    def __init__(self, ckpt: str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import StableDiffusionPipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(ckpt, precision=precision).to(device)
        self.device = device
    def editimage(self, prompt, image, num_iters=200):
        if isinstance(image, Image.Image):
            # img to numpy array
            image = np.array(image)
        if isinstance(prompt, str):
            raise ValueError("Prompt should be a dictionary with 'source_prompt' and 'target_prompt' keys")
        text_source = prompt["source_prompt"]
        text_target = prompt["target_prompt"]
        
        from DDS_zeroshot_utils import get_text_embeddings, DDSLoss, decode
        from torch.optim.adamw import AdamW
        from torch.optim.sgd import SGD
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel
        dds_loss = DDSLoss(self.device, self.pipeline)
        image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
        image_source = image_source.unsqueeze(0).to(self.device)
        with torch.no_grad():
            z_source = self.pipeline.vae.encode(image_source)['latent_dist'].mean * 0.18215
            image_target = image_source.clone()
            embedding_null = get_text_embeddings(self.pipeline, "")
            embedding_text = get_text_embeddings(self.pipeline, text_source)
            embedding_text_target = get_text_embeddings(self.pipeline, text_target)
            embedding_source = torch.stack([embedding_null, embedding_text], dim=1)
            embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)
        guidance_scale = 7.5
        image_target.requires_grad = True
        use_dds_loss = True

        z_taregt = z_source.clone()
        z_taregt.requires_grad = True
        optimizer = SGD(params=[z_taregt], lr=1e-1)
        # num_iters = kwargs.get("num_iters", 200)
        # use_dds = kwargs.get("use_dds", True)
        for i in range(num_iters):
            loss, log_loss = dds_loss.get_dds_loss(z_source, z_taregt, embedding_source, embedding_target)
            optimizer.zero_grad()
            (2000 * loss).backward()
            optimizer.step()
            
        out = decode(z_taregt, self.pipeline, im_cat=None)
        return out
    
    
