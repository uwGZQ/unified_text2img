import sys
# sys.path.append('VidToMe')  
# sys.path.append('RAVE')
# sys.path.append('flatten')
import os
import argparse
import torch
import torchvision
from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL, DDIMInverseScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# from models.pipeline_flatten import FlattenPipeline
# from models.util import save_videos_grid, read_video, sample_trajectories
# from models.unet import UNet3DConditionModel
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'RAVE'))) 
# sys.path.append(os.getcwd())


import itertools
import yaml
import datetime
import numpy as np


videoedit_model = {
    "VidtoMe": ("VidtoMe_model",  "runwayml/stable-diffusion-v1-5"),
    "RAVE": ("RAVE_model", "runwayml/stable-diffusion-v1-5"),
    "Flatten": ("Flatten_model", ["stabilityai/stable-diffusion-2-1-base","flatten/checkpoints/stable-diffusion-2-1-base/unet"])

}

def set_model_key(model_name, key):
	videoedit_model[model_name] = (videoedit_model[model_name][0], key)

def list_videoedit_models():
    return list(videoedit_model.keys())

class AbstractModel:
    def videoedit(self, prompt, vid_path):
        "(Abstract method) abstract video_edit method"

class VideoEdit:
    def __init__(self, model_name: str = "VidtoMe", model: AbstractModel = None, ckpt: str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, torch_device: str = "cuda"):
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
            class_name, ckpt = videoedit_model[model_name]
            self.model_presision = precision
            self.model = eval(class_name)(ckpt, precision, torch_device)
            print(f"Finish loading {model_name}")
        else:
            print(f"Using provided model ...")
            class_name = model.__class__.__name__
            self.model = eval(class_name)(ckpt, precision, torch_device)
    
    def tensor_to_pil_images(self, tensor):
        from PIL import Image
        if tensor.dim() != 5 or tensor.shape[1] != 3:
            raise ValueError("Tensor should be of shape [batch_size, channels, frames, height, width]")

        batch_images = []

        for batch in range(tensor.size(0)):  
            pil_images = []
            for i in range(tensor.size(2)):  
                frame = tensor[batch, :, i, :, :]
                frame = frame.mul(255).byte()
                frame = frame.permute(1, 2, 0).numpy()
                pil_image = Image.fromarray(frame)
                pil_images.append(pil_image)
            batch_images.append(pil_images)

        return batch_images[0]
    
    def video_to_pil_images(self, video_path):
        import cv2
        from PIL import Image
        import numpy as np
        cap = cv2.VideoCapture(video_path)
        pil_images = []

        if not cap.isOpened():
            print("Error opening video file")
            return []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            pil_images.append(pil_image)

        cap.release()

        return pil_images
    @torch.no_grad()
    def editvideo(self, prompt, vid_path, **kwargs):
        result =  self.model.editvideo(prompt, vid_path, **kwargs)
        if isinstance(result, str):
            return self.video_to_pil_images(result)
        elif isinstance(result, list):
            return result
        elif isinstance(result, torch.Tensor):
            return self.tensor_to_pil_images(result)
            

class VidtoMe_model():
    def __init__(self, ckpt: str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        sys.path.append('VidToMe')  

        from utils import load_config, seed_everything
        self.config = load_config(path = "VidToMe/configs/default.yaml", print_config=False)
        if precision == torch.float16:
            precision = "fp16"
        else:
            precision = "fp32"
        
        self.ckpt = ckpt
        self.precision = precision
        self.device = device
        seed_everything(self.config.seed)
    def editvideo(self, prompt, vid_path, control = "depth",**kwargs):
        sys.path.append('VidToMe')  

        from invert import Inverter
        from generate import Generator
        from utils import get_frame_ids, init_model
        from omegaconf import OmegaConf
        # def extract_file_name(input_path):
        #     return input_path.split('/')[-1].split('.')[0]
        
        # self.config.work_dir = self.config.work_dir+ extract_file_name(vid_path)
        from torchvision.io import read_video
        num_frames = read_video(vid_path, output_format="TCHW", pts_unit="sec")[0].shape[0]
        self.config.generation.frame_range = [0,num_frames,1]
        
        self.config.input_path = vid_path
        
        self.config.work_dir = self.config.work_dir +"/"+ vid_path.split('/')[-1].split('.')[0]
        self.config.inversion.save_path = self.config.work_dir + '/latents'
        self.config.generation.latents_path = self.config.work_dir + '/latents'
        self.config.generation.output_path = self.config.work_dir
        self.config.generation.control = control
        
        for key, value in kwargs.items():
            if hasattr(self.config.generation, key):
                setattr(self.config.generation, key, value)
            elif hasattr(self.config.inversion, key):
                setattr(self.config.inversion, key, value)
        
        if isinstance(prompt, dict):
            text_source = prompt.get("text_source", "")
            text_target = prompt.get("text_target")
        elif isinstance(prompt, str):
            text_source = ""
            text_target = prompt

        
        self.config.inversion.prompt = text_source
        self.config.generation.prompt =  {"edit": text_target}


        
        self.pipe, self.scheduler, self.model_key = init_model(
        "cuda",  model_key = self.ckpt, control_type = self.config.generation.control, weight_dtype = self.precision)
        self.config.model_key = self.model_key
        
        inversion = Inverter(self.pipe, self.scheduler, self.config)
        inversion(self.config.input_path, self.config.inversion.save_path)
        
        generator = Generator(self.pipe, self.scheduler, self.config)
        frame_ids = get_frame_ids(
            self.config.generation.frame_range, self.config.generation.frame_ids)
        generator(self.config.input_path, self.config.generation.latents_path,
                self.config.generation.output_path, frame_ids=frame_ids)
        # return all the frames of the edited video
        # frames are in output_path/frames
        vid_path = os.path.join(self.config.generation.output_path,"edit" ,"output.mp4")
        return vid_path
        




class RAVE_model(AbstractModel):
    def __init__(self, ckpt: str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        # config_path = "RAVE/configs/truck.yaml"
        config_path = "RAVE/configs/truck.yaml"
        self.input_dict_list = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.device = device
        self.ckpt = ckpt
    
    def editvideo(self, prompt, vid_path, preprocess_name = "depth_zoe"):
        if isinstance(prompt, dict):
            prompt = prompt.get("text_target")
        elif isinstance(prompt, str):
            prompt = prompt
        self.input_dict_list['positive_prompts'] = prompt
        self.input_dict_list['video_name'] = vid_path
        self.input_dict_list['save_folder'] = vid_path.split('/')[-1].split('.')[0]
        self.input_dict_list['preprocess_name'] = preprocess_name
        input_ns = argparse.Namespace(**self.input_dict_list)
        return self.run(input_ns)
        
       
    def init_device(self):
        return self.device

    def init_paths(self, input_ns):
        sys.path.append('RAVE')
        import RAVE.utils.constants as const
        from datetime import datetime 


        if input_ns.save_folder == None or input_ns.save_folder == '':
            input_ns.save_folder = input_ns.video_name.split('/')[-1].split('.')[0]
        # else:
        #     input_ns.save_folder += f"/{input_ns.video_name.replace('.mp4', '').replace('.gif', '')}"
        # save_dir = f'{const.OUTPUT_PATH}/{input_ns.save_folder}'
        save_dir = os.path.join("RAVE_workdir","results", datetime.now().strftime("%m-%d-%Y"),input_ns.save_folder)
        
        os.makedirs(save_dir, exist_ok=True)
        
        save_idx = max([int(x[-5:]) for x in os.listdir(save_dir)])+1 if os.listdir(save_dir) != [] else 0
        input_ns.save_path = f'{save_dir}/{input_ns.positive_prompts}-{str(save_idx).zfill(5)}'
        
        input_ns.video_path = input_ns.video_name
        # input_ns.video_path = f'{const.MP4_PATH}/{input_ns.video_name}.mp4'
        
        if '-' in input_ns.preprocess_name:
            input_ns.hf_cn_path = [const.PREPROCESSOR_DICT[i] for i in input_ns.preprocess_name.split('-')]
        else:
            input_ns.hf_cn_path = const.PREPROCESSOR_DICT[input_ns.preprocess_name]
        
        input_ns.hf_path = self.ckpt
        
        input_ns.inverse_path = f'{const.GENERATED_DATA_PATH}/inverses/{input_ns.video_name}/{input_ns.preprocess_name}_{input_ns.model_id}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}'
        input_ns.control_path = f'{const.GENERATED_DATA_PATH}/controls/{input_ns.video_name}/{input_ns.preprocess_name}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}'
        os.makedirs(input_ns.control_path, exist_ok=True)
        os.makedirs(input_ns.inverse_path, exist_ok=True)
        os.makedirs(input_ns.save_path, exist_ok=True)
        
        return input_ns
        
    def run(self, input_ns):
        sys.path.append('RAVE')
        from RAVE.pipelines.sd_controlnet_rave import RAVE
        from RAVE.pipelines.sd_multicontrolnet_rave import RAVE_MultiControlNet
        import RAVE.utils.constants as const
        import RAVE.utils.video_grid_utils as vgu


        if 'model_id' not in list(input_ns.__dict__.keys()):
            input_ns.model_id = "None"
        device = self.init_device()
        input_ns = self.init_paths(input_ns)
        
        # print(input_ns)

        input_ns.image_pil_list = vgu.prepare_video_to_grid(input_ns.video_path, input_ns.sample_size, input_ns.grid_size, input_ns.pad)
        
        input_ns.sample_size = len(input_ns.image_pil_list)
        print(f'Frame count: {len(input_ns.image_pil_list)}')

        controlnet_class = RAVE_MultiControlNet if '-' in str(input_ns.controlnet_conditioning_scale) else RAVE
        

        print(input_ns)
        CN = controlnet_class(device)


        CN.init_models(input_ns.hf_cn_path, input_ns.hf_path, input_ns.preprocess_name, input_ns.model_id)
        
        input_dict = vars(input_ns)
        yaml_dict = {k:v for k,v in input_dict.items() if k != 'image_pil_list'}

        start_time = datetime.datetime.now()
        if '-' in str(input_ns.controlnet_conditioning_scale):
            res_vid, control_vid_1, control_vid_2 = CN(input_dict)
        else: 
            res_vid, control_vid = CN(input_dict)
        end_time = datetime.datetime.now()
        save_name = f"{'-'.join(input_ns.positive_prompts.split())}_cstart-{input_ns.controlnet_guidance_start}_gs-{input_ns.guidance_scale}_pre-{'-'.join((input_ns.preprocess_name.replace('-','+').split('_')))}_cscale-{input_ns.controlnet_conditioning_scale}_grid-{input_ns.grid_size}_pad-{input_ns.pad}_model-{input_ns.model_id.split('/')[-1]}"
        res_vid[0].save(f"{input_ns.save_path}/{save_name}.gif", save_all=True, append_images=res_vid[1:], optimize=False, loop=10000)
        # control_vid[0].save(f"{input_ns.save_path}/control_{save_name}.gif", save_all=True, append_images=control_vid[1:], optimize=False, loop=10000)

        yaml_dict['total_time'] = (end_time - start_time).total_seconds()
        yaml_dict['total_number_of_frames'] = len(res_vid)
        yaml_dict['sec_per_frame'] = yaml_dict['total_time']/yaml_dict['total_number_of_frames']
        with open(f'{input_ns.save_path}/config.yaml', 'w') as yaml_file:
            yaml.dump(yaml_dict, yaml_file)
        
        return res_vid
     
     


class Flatten_model(AbstractModel):
    def __init__(self, ckpt : list = ["stabilityai/stable-diffusion-2-1-base","flatten/checkpoints/stable-diffusion-2-1-base/unet"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        sys.path.append('flatten')
        from models.pipeline_flatten import FlattenPipeline
        from models.unet import UNet3DConditionModel
        self.output_path = "flatten_workdir"
        os.makedirs("flatten_workdir", exist_ok=True)
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(ckpt[0], subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(ckpt[0], subfolder="text_encoder").to(dtype=torch.float16)
        self.vae = AutoencoderKL.from_pretrained(ckpt[0], subfolder="vae").to(dtype=torch.float16)
        self.unet = UNet3DConditionModel.from_pretrained_2d(ckpt[0], subfolder="unet").to(dtype=torch.float16)
        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(ckpt[1], WEIGHTS_NAME)

        state_dict = torch.load(model_file, map_location="cpu")
        self.unet.load_state_dict(state_dict, strict=False)
        
        self.scheduler=DDIMScheduler.from_pretrained(ckpt[0], subfolder="scheduler")
        self.inverse=DDIMInverseScheduler.from_pretrained(ckpt[0], subfolder="scheduler")

        self.pipe = FlattenPipeline(
                vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
                scheduler=self.scheduler, inverse_scheduler=self.inverse)
        self.pipe.enable_vae_slicing()
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.to(device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(66)
        
    def editvideo(self, prompt, vid_path, height = 512, width = 512,  video_length = 8, guidance_scale = 20.0, sample_steps = 50, inject_step = 40, old_qk = 0, fps = 15):
        sys.path.append('flatten')

        from models.util import save_videos_grid, read_video, sample_trajectories

        if isinstance(prompt, dict):
            tgt_prompt = prompt.get("text_target")
            neg_prompt = prompt.get("neg_prompt", "")
        elif isinstance(prompt, str):
            tgt_prompt = prompt
            neg_prompt = ""
        
        self.height = height
        self.width = width
        
        video = read_video(video_path=vid_path, video_length=video_length)
        original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
        save_videos_grid(original_pixels, os.path.join("flatten_workdir",self.sanitize_filename(tgt_prompt), "source_video.mp4"), rescale=True)

        t2i_transform = torchvision.transforms.ToPILImage()
        real_frames = []
        for i, frame in enumerate(video):
            real_frames.append(t2i_transform(((frame+1)/2*255).to(torch.uint8)))

        # compute optical flows and sample trajectories
        trajectories = sample_trajectories(os.path.join(self.output_path, self.sanitize_filename(tgt_prompt), "source_video.mp4"), self.device)
        torch.cuda.empty_cache()

        for k in trajectories.keys():
            trajectories[k] = trajectories[k].to(self.device)
        

        
        sample = self.pipe(tgt_prompt, video_length=video_length, frames=real_frames,
                    num_inference_steps=sample_steps, generator=self.generator, guidance_scale=guidance_scale,
                    negative_prompt=neg_prompt, width=self.width, height=self.height,
                    trajs=trajectories, output_dir="tmp/", inject_step=40, old_qk=old_qk).videos
        temp_video_name = tgt_prompt+"_"+neg_prompt+"_"+str(guidance_scale)
        save_videos_grid(sample, f"{self.output_path}/{self.sanitize_filename(tgt_prompt)}/{temp_video_name}.mp4", fps=fps)
        return sample
    
    
    def sanitize_filename(self, prompt):
        import re
        sanitized = re.sub(r'[\\/*?:"<>|]', '', prompt)
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized.strip('.-_')
        return sanitized
        





