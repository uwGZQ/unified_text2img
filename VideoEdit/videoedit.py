
import sys
import os
sys.path.append('VidToMe')  
sys.path.append('RAVE')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'RAVE'))) 
# sys.path.append(os.getcwd())

import torch
import argparse

import itertools
import sys
import yaml
import datetime
import numpy as np

# RAVE
# from pipelines.sd_controlnet_rave import RAVE
# from pipelines.sd_multicontrolnet_rave import RAVE_MultiControlNet

# import utils.constants as const
# import utils.video_grid_utils as vgu

# import warnings

# warnings.filterwarnings("ignore")




class VidtoMe():
    def __init__(self, ckpt: str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from utils import init_model, load_config, seed_everything
        self.config = load_config(path = "VidToMe/configs/default.yaml", print_config=False)
        if precision == torch.float16:
            precision = "fp16"
        else:
            precision = "fp32"
        self.pipe, self.scheduler, self.model_key = init_model(
        "cuda",  model_key = ckpt, control_type = self.config.generation.control, weight_dtype = precision)
        self.config.model_key = self.model_key
        seed_everything(self.config.seed)
    def editvideo(self, prompt, vid_path):
        from invert import Inverter
        from generate import Generator
        from utils import get_frame_ids
        # def extract_file_name(input_path):
        #     return input_path.split('/')[-1].split('.')[0]
        
        # self.config.work_dir = self.config.work_dir+ extract_file_name(vid_path)
        from torchvision.io import read_video
        num_frames = read_video(vid_path, output_format="TCHW", pts_unit="sec")[0].shape[0]
        self.config.generation.frame_range = [0,num_frames,1]
        
        self.config.input_path = vid_path
        text_source = prompt.get("text_source", "")
        text_target = prompt.get("text_target")
        
        self.config.inversion.prompt = text_source
        self.config.generation.prompt = {"target":text_target}
        
        print(self.config)
        inversion = Inverter(self.pipe, self.scheduler, self.config)
        inversion(self.config.input_path, self.config.inversion.save_path)
        
        generator = Generator(self.pipe, self.scheduler, self.config)
        frame_ids = get_frame_ids(
            self.config.generation.frame_range, self.config.generation.frame_ids)
        generator(self.config.input_path, self.config.generation.latents_path,
                self.config.generation.output_path, frame_ids=frame_ids)


# model =  VidtoMe()
# model.editvideo({"text_source": "a panda", "text_target": "A beautiful sunrise"}, "/data1/ziqi/unified_text2vid/video.mp4", "output.mp4")        



class RAVE_model():
    def __init__(self, ckpt: str = "depth_zoe", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        # config_path = "RAVE/configs/truck.yaml"
        config_path = "RAVE/configs/truck.yaml"
        self.input_dict_list = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.device = device
    
    def editvideo(self, prompt, vid_path):
        self.input_dict_list['positive_prompts'] = prompt
        self.input_dict_list['video_name'] = vid_path
        input_ns = argparse.Namespace(**self.input_dict_list)
        self.run(input_ns)
        
       
    def init_device(self):
        # device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = torch.device(device_name)
        return self.device

    def init_paths(self, input_ns):
        import RAVE.utils.constants as const


        if input_ns.save_folder == None or input_ns.save_folder == '':
            input_ns.save_folder = input_ns.video_name.replace('.mp4', '').replace('.gif', '')
        else:
            input_ns.save_folder += f"/{input_ns.video_name.replace('.mp4', '').replace('.gif', '')}"
        save_dir = f'{const.OUTPUT_PATH}/{input_ns.save_folder}'
        os.makedirs(save_dir, exist_ok=True)
        
        save_idx = max([int(x[-5:]) for x in os.listdir(save_dir)])+1 if os.listdir(save_dir) != [] else 0
        input_ns.save_path = f'{save_dir}/{input_ns.positive_prompts}-{str(save_idx).zfill(5)}'
        

        input_ns.video_path = f'{const.MP4_PATH}/{input_ns.video_name}.mp4'
        
        if '-' in input_ns.preprocess_name:
            input_ns.hf_cn_path = [const.PREPROCESSOR_DICT[i] for i in input_ns.preprocess_name.split('-')]
        else:
            input_ns.hf_cn_path = const.PREPROCESSOR_DICT[input_ns.preprocess_name]
        input_ns.hf_path = "runwayml/stable-diffusion-v1-5"
        
        input_ns.inverse_path = f'{const.GENERATED_DATA_PATH}/inverses/{input_ns.video_name}/{input_ns.preprocess_name}_{input_ns.model_id}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}'
        input_ns.control_path = f'{const.GENERATED_DATA_PATH}/controls/{input_ns.video_name}/{input_ns.preprocess_name}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}'
        os.makedirs(input_ns.control_path, exist_ok=True)
        os.makedirs(input_ns.inverse_path, exist_ok=True)
        os.makedirs(input_ns.save_path, exist_ok=True)
        
        return input_ns
        
    def run(self, input_ns):
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
        
     
     

        
# model = RAVE_model(
    
# )
# model.editvideo("Wooden trucks drive on a racetrack", "truck")