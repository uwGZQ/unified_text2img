# Unified Text-to-Image / Video / 3D Generation, Image / Video Editting and Evaluation Metrics

This repository provides a unified interface for generating images / videos from text prompts using various state-of-the-art models including DALL-E, Stable Diffusion, and many others. It's designed to simplify the process of integrating and switching between different text-to-image/videos models.

## Setup

Clone the repository and set up a Python environment:

```bash
git clone https://github.com/your-repo/unified_text2img.git
cd unified_text2img
python3.12 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt
```


## Using the Library
To generate an image from a text prompt, use the following code snippet:
```bash
from text2img_models import Text2Img

# Initialize with the model of your choice
model = Text2Img(model_name="stable-diffusion-2-1")

# Generate an image from a prompt
prompt = "A panoramic view of a futuristic city at sunset"
img = model.text2img(prompt)

# Save the generated image
img.save("generated_image.jpg")

```
Note that the inputs required by the text2img method may vary depending on the model selected. Be sure to adjust the method's arguments accordingly based on the specific requirements of each model.

Additionally, some models may require authentication with Hugging Face to download and use them. To authenticate, use the following command in your terminal:

```bash
huggingface-cli login
```
Enter your Hugging Face API key when prompted. This will store the token locally and allow you to access models from Hugging Face within your application.

To generate a video from a text prompt, use the following code snippet:
```bash
from text2vid_models

import os

model_name = 'text2vid-zero'

prompt = "A man is playing VR Games in his house."

save_directory = "./generated_videos/"

model = Text2Vid(model_name=model_name)

video_frames = model.text2vid(prompt)

model.save_vid(video_frames, os.path.join(save_directory, f"{model_name}.mp4"), fps=4)
```

To evaluate your result, using the following example:
```bash
from Eval_metrics import Compute_Metrics
metric_name = "InceptionScore"
metric = Compute_Metrics(metric_name = metric_name)

imgs_folder = "./images"

metric.update(imgs = imgs_folder)

result = metric.compute()

```


When adding a new model, ensure to include any additional dependencies in `requirements.txt` and make a pull request.

## Supported Models

The library currently supports a wide range of models, including but not limited to:

text2img:
* DALL-E 2 & 3
* Stable Diffusion (versions 1.4 to 2.1, including XL and Safe variants)
* Kandinsky (versions 2.1, 2.2, and 3)
* PixArt
* Multi-Diffusion
* DeepFloyd
* Wuerstchen
* GLIGEN
* stable-cascade
* playground-v2
* ldm3d
* sdxl-lightning
* SSD-1B
* sdxl-turbo
* stable-unclip
* DiT
* latent-consistency-model
* LDM

Text2vid:

* Text2vid-Zero
* Zeroscope
* Modelscope-t2v
* Animatediff
* Animatediff-motion-lora    
* AnimateDif_motion_lora_peft
* AnimateLCM
* AnimateLCM-motion-lora 
* FreeInit

Text-to-3D:
* LDM3D
* LDM3D_4C
* LDM3D_Pano
* ShapE
* Dreamfusion_sd
* Dreamfusion_if
* Prolificdreamer
* Magic3D_if
* Magic3d-sd
* SJC
* LatentNeRF
* Fantasia3D
* TextMesh
* ProlificDreamer_HiFA
* DreamFusion_HiFA

Image Edit:
* InstructPix2Pix
* LEditsPP_sd
* LEditsPP_sdxl
* LEDITS
* Pix2PixZero
* DDIM
* PromptToPromptDDIM
* PromptToPromptInversion
* DDPMInversion
* ScoreDistillationSampling
* DDS_zero_shot

Video Edit:
* VidtoMe
* RAVE
* Flatten


Metrics:
* InceptionScore
* FrechetInceptionDistance
* LPIPS
* KernelInceptionDistance
* ClipScore
* HPSv2
* PickScore_v1
* ImageReward
* Clip-R-Precision
* SemanticObjectAccuracy
* MutualInformationDivergence
* Vbench


## Extending the Library

To add a new model:

1. Implement a new class for the model inheriting from `AbstractModel`.
2. Define the `text2img` \ `text2vid` \ `Compute_Metrics` method according to the model's requirements.
3. Update the `text2img_model` \ `text2vid_model` \ `Eval_metrics` dictionary with the new model's details.
4. Ensure to add any model-specific dependencies to `requirements.txt`.

## Contributing

Contributions are welcome! If you add a new model or make improvements, please submit a pull request.

## Issues & Support

For bugs, issues, or feature requests, please file an issue through the GitHub issue tracker.

