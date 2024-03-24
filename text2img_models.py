from typing import Union

import diskcache
import openai
import torch
from PIL import Image
from openai import OpenAI

# DALLE, Stable Diffusion
text2img_model = {
	"dalle3"                            : ("DALLEModel", {"model": "dall-e-3", "api-key": "sk-", "quality": "standard", "size": "1024x1024", "cache_path": './dalle3_cache'},),
	"dalle2"                            : ("DALLEModel", {"model": "dall-e-2", "api-key": "sk-", "size": "1024x1024", "cache_path": './dalle2_cache'},),
	"stable-diffusion-2-1"              : ('StableDiffusion2', "stabilityai/stable-diffusion-2-1",),
	"stable-diffusion-1-5"              : ('StableDiffusion', "runwayml/stable-diffusion-v1-5",),
	"stable-diffusion-1-4"              : ('StableDiffusion', "CompVis/stable-diffusion-v1-4",),
	"stable-diffusion-xl"               : ("StableDiffusionXL", "stabilityai/stable-diffusion-xl-base-1.0",),
	"stable-diffusion-safe"             : ("StableDiffusionSafe", "AIML-TUDA/stable-diffusion-safe",),
	"stable-cascade"                    : ("StableCascade", {"prior-pipeline": "stabilityai/stable-cascade-prior", "pipeline": "stabilityai/stable-cascade"},),
	"sdxl-turbo"                        : ("SDXLTurbo", "stabilityai/sdxl-turbo",),
	"bk-sdm"                            : ("BK_SDM", "segmind/SSD-1B",),
	"wuerstchen"                        : ("Wuerstchen", "warp-ai/Wuerstchen",),
	"stable-diffusion-attend-and-excite": ("AttendAndExcite", "CompVis/stable-diffusion-v1-4",),
	"deepfloyd-if"                      : ("DeepFloyd", {"pipeline": ["DeepFloyd/IF-I-XL-v1.0", "DeepFloyd/IF-II-L-v1.0", "stabilityai/stable-diffusion-x4-upscaler"]},),
	"kandinsky-2-1"                     : ("KandinskyV21", {"prior-pipeline": "kandinsky-community/kandinsky-2-1-prior", "pipeline": 'kandinsky-community/kandinsky-2-1'},),
	"kandinsky-2-2"                     : ("KandinskyV22", {"prior-pipeline": "kandinsky-community/kandinsky-2-2-prior", "pipeline": 'kandinsky-community/kandinsky-2-2-decoder'},),
	"kandinsky-3"                       : ("KandinskyV3", "kandinsky-community/kandinsky-3",),
	"playground-v2"                     : ("Playgroundv2", "playgroundai/playground-v2-1024px-aesthetic",),
	"self-attention-guidance"           : ("SelfAttentionGuidance", "runwayml/stable-diffusion-v1-5",),
	"gligen-text-image"                 : ("GLIGEN", "anhnct/Gligen_Text_Image",),
	"ldm3d"                             : ("LDM3D", "Intel/ldm3d-4c",),
	# "alt-diffusion": ("AltDiffusion","BAAI/AltDiffusion-m9",),
	"stable-unclip"                     : ("StableUnClip", {"prior_model": "kakaobrain/karlo-v1-alpha", "prior_text_model": "openai/clip-vit-large-patch14", "pipeline": "stabilityai/stable-diffusion-2-1-unclip-small"},),
	"sdxl-lightning"                    : ("SDXLLightning", {"repo": "ByteDance/SDXL-Lightning", "base": "stabilityai/stable-diffusion-xl-base-1.0", "step": 4, "_ckpt": "sdxl_lightning_4step_unet.safetensors", "use_unet": True},),
	"latent-consistency-model"          : ("LatentConsistencyModel", "SimianLuo/LCM_Dreamshaper_v7",),
	"ldm-text2im-large-256"             : ("LDM", "CompVis/ldm-text2im-large-256",),
	"multi-diffusion"                   : ("MultiDiffusion", "stabilityai/stable-diffusion-2-base",),
	"dit"                               : ("DiT", "facebook/DiT-XL-2-256",),
	"pixart"                            : ("PixArt", "PixArt-alpha/PixArt-XL-2-1024-MS",),
}


def set_model_key(model_name, key):
	text2img_model[model_name] = (text2img_model[model_name][0], key)


def list_text2img_models():
	return list(text2img_model.keys())


class AbstractModel:
	def text2img(self, prompt):
		"(Abstract method) abstract text2img method"


class Text2Img:
	def __init__(self, model_name: str = "stable-diffusion-2-1", model: AbstractModel = None, ckpt: str = "stabilityai/stable-diffusion-2-1", precision: torch.dtype = torch.float16, torch_device: str = "cuda", calculate_metrics: bool = False):
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
			class_name, ckpt = text2img_model[model_name]
			self.model_presision = precision
			self.model = eval(class_name)(ckpt, precision, torch_device)
			print(f"Finish loading {model_name}")
		else:
			print(f"Using provided model ...")
			class_name = model.__class__.__name__
			self.model = eval(class_name)(ckpt, precision, torch_device)
		self.calculate_metrics = calculate_metrics

	@torch.no_grad()
	def text2img(self, prompt):
		img = self.model.text2img(prompt)
		if self.calculate_metrics:
			clip_score = self.Clip_metrics(prompt, img)
			return img[0], clip_score
		return img[0]

	@torch.no_grad()
	def text2img_indices(self, prompt, active_indices=[0]):
		if self.model.__class__.__name__ == "AttendAndExcite":
			img = self.model.text2img(prompt, active_indices)
		else:
			img = self.model.text2img(prompt)
		if self.calculate_metrics:
			clip_score = self.Clip_metrics(prompt, img)
			return img[0], clip_score
		return img[0]

	@torch.no_grad()
	def text2img_boxes(self, prompt, boxes, phrases):
		if self.model.__class__.__name__ == "GLIGEN":
			img = self.model.text2img(prompt, boxes, phrases)
		else:
			img = self.model.text2img(prompt)
		if self.calculate_metrics:
			clip_score = self.Clip_metrics(prompt, img)
			return img[0], clip_score
		return img[0]

	@torch.no_grad()
	def get_prompt_indices(self, prompt):
		if self.model.__class__.__name__ == "AttendAndExcite":
			return self.model.get_prompt_indices(prompt)
		else:
			# raise warning
			print("This model does not support get_prompt_indices method.")
			return prompt.split()

	@torch.no_grad()
	def save_img(self, img, path):
		img.save(path)


class DALLEModel(AbstractModel):
	def __init__(self, ckpt: dict = {"model": "dall-e-3", "api-key": "sk-", "quality": "standard", "size": "1024x1024", "cache_path": './dalle3_cache'}, precision=torch.float16, device=torch.device("cuda")):
		if isinstance(ckpt["api-key"], str):
			self.client = OpenAI(api_key=ckpt["api-key"])
		elif isinstance(ckpt["api-key"], list):
			self.client = [OpenAI(api_key=c) for c in ckpt["api-key"]]
		else:
			raise ValueError("Invalid API Key")

		# Must be one of 256x256, 512x512, or 1024x1024 for dall-e-2. Must be one of 1024x1024, 1792x1024, or 1024x1792 for dall-e-3 models.
		if ckpt["model"] == 'dall-e-2':
			assert ckpt["size"] in ['256x256', '512x512', '1024x1024']
		elif ckpt["model"] == 'dall-e-3':
			assert ckpt["quality"] in ['standard', 'hd']
			assert ckpt["size"] in ['1024x1024', '1792x1024', '1024x1792']
		self.model = ckpt["model"]
		self.completion_images = 0
		self.cache_path = ckpt['cache_path']
		self.size = ckpt["size"]
		self.price = 0.0
		if self.model == "dall-e-3":
			self.quality = ckpt["quality"]
		else:
			self.quality = None

	def _get_response(self, client, prompt):
		with diskcache.Cache(self.cache_path) as cache:
			key = str(prompt)
			response = cache.get(key)
			print(response)
			if response is None:
				while True:
					try:
						if self.model == "dall-e-2":
							response = client.images.generate(model=self.model,
															  prompt=prompt,
															  size=self.size,
															  response_format="b64_json",
															  n=1,
															  )
						else:
							response = client.images.generate(model=self.model,
															  prompt=prompt,
															  size=self.size,
															  quality=self.quality,
															  response_format="b64_json",
															  n=1,
															  )
					except openai.OpenAIError as e:
						if e.code == "sanitizer_server_error":
							continue
						else:
							print(e)
							raise e
					break
				print(response)
				cache.set(key, response)
		return response

	def get_cost(self):
		return self.price

	def _cost(self, n, quality, size, model):
		if model == "dall-e-3":
			if quality == "standard":
				if size == "1024x1024":
					return 0.04 * n
				else:
					return 0.08 * n
			else:
				if size == "1024x1024":
					return 0.08 * n
				else:
					return 0.12 * n
		elif model == "dall-e-2":
			if size == "256x256":
				return 0.016 * n
			elif size == "512x512":
				return 0.018 * n
			else:
				return 0.02 * n

	def text2img(self, prompt):
		if isinstance(self.client, list):
			pointer = 0
			client = self.client[pointer]
			try:
				response = self._get_response(client, prompt)
			except openai.RateLimitError as e:
				if pointer < len(self.client) - 1:
					pointer += 1
					client = self.client[pointer]
					response = self._get_response(client, prompt)
				else:
					raise e
		else:
			response = self._get_response(self.client, prompt)
		self.completion_images += len(response.data)
		self.price += self._cost(len(response.data), self.quality, self.size, self.model)
		import base64
		from io import BytesIO
		decoded_bytes_list = [base64.b64decode(response.data[i].b64_json) for i in range(len(response.data))]
		img = [Image.open(BytesIO(decoded_bytes_list[i])) for i in range(len(response.data))]
		# img = [Image.open(requests.get(url, stream=True).raw) for url in img_url]
		return img


class StableDiffusion(AbstractModel):
	def __init__(self, ckpt: str = 'runwayml/stable-diffusion-v1-5', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import StableDiffusionPipeline
		self.pipeline = StableDiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)

	def text2img(self, prompt):
		return self.pipeline(prompt).images


class StableDiffusion2(AbstractModel):
	def __init__(self, ckpt: str = 'stabilityai/stable-diffusion-2-base', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
		self.pipeline = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision, revision="fp16")
		self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
		self.pipeline = self.pipeline.to(device)

	def text2img(self, prompt):
		return self.pipeline(prompt, num_inference_steps=25).images


class StableDiffusionXL(AbstractModel):
	def __init__(self, ckpt: str = 'stabilityai/stable-diffusion-xl-base-1.0', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import StableDiffusionXLPipeline
		self.pipeline = StableDiffusionXLPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)

	def text2img(self, prompt):
		return self.pipeline(prompt).images


class StableDiffusionSafe(AbstractModel):
	def __init__(self, ckpt: str = "AIML-TUDA/stable-diffusion-safe", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import StableDiffusionPipeline
		self.pipeline = StableDiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)

	def text2img(self, prompt):
		from diffusers.pipelines.stable_diffusion_safe import SafetyConfig

		return self.pipeline(prompt=prompt, **SafetyConfig.MEDIUM).images


class SDXLTurbo(AbstractModel):
	def __init__(self, ckpt: str = "stabilityai/sdxl-turbo", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import AutoPipelineForText2Image
		self.pipeline = AutoPipelineForText2Image.from_pretrained(ckpt, torch_dtype=precision, variant="fp16").to(device)

	def text2img(self, prompt):
		return self.pipeline(prompt=prompt).images


class BK_SDM(AbstractModel):
	def __init__(self, ckpt: str = "segmind/SSD-1B", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import StableDiffusionXLPipeline
		self.pipeline = StableDiffusionXLPipeline.from_pretrained(ckpt, torch_dtype=precision, use_safetensors=True, variant="fp16").to(device)

	def text2img(self, prompt):
		return self.pipeline(prompt=prompt).images


class StableCascade(AbstractModel):
	def __init__(self, ckpt: dict = {"prior-pipeline": "stabilityai/stable-cascade-prior", "pipeline": "stabilityai/stable-cascade"}, precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		assert len(ckpt) == 2 and isinstance(ckpt, dict) and "prior-pipeline" in ckpt and "pipeline" in ckpt

		from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

		self.pipeline = StableCascadeDecoderPipeline.from_pretrained(ckpt["pipeline"], torch_dtype=precision).to(device)
		self.pipe_prior = StableCascadePriorPipeline.from_pretrained(ckpt["prior-pipeline"], torch_dtype=torch.bfloat16).to(device)
		self.pipe_prior.enable_model_cpu_offload()
		self.pipeline.enable_model_cpu_offload()

	def text2img(self, prompt):
		prior_output = self.pipe_prior(
			prompt=prompt,
			height=1024,
			width=1024,
			# negative_prompt=negative_prompt,
			guidance_scale=4.0,
			num_images_per_prompt=1,
			num_inference_steps=20
		)
		return self.pipeline(
			image_embeddings=prior_output.image_embeddings.to(torch.float16),
			prompt=prompt,
			# negative_prompt=negative_prompt,
			guidance_scale=0.0,
			output_type="pil",
			num_inference_steps=10
		).images


class Wuerstchen(AbstractModel):
	def __init__(self, ckpt: str = 'warp-ai/Wuerstchen', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import WuerstchenCombinedPipeline
		self.pipeline = WuerstchenCombinedPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)

	def text2img(self, prompt):
		return self.pipeline(prompt=prompt).images


# need token_indices
class AttendAndExcite(AbstractModel):
	def __init__(self, ckpt: str = 'CompVis/stable-diffusion-v1-4', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import StableDiffusionAttendAndExcitePipeline
		self.pipeline = StableDiffusionAttendAndExcitePipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)

	def get_prompt_indices(self, prompt):
		return self.pipeline.get_indices(prompt)

	def text2img(self, prompt, token_indices: Union[list, tuple] = None):
		assert token_indices is not None
		seed = 6141
		generator = torch.Generator(device='cuda').manual_seed(seed)
		images = self.pipeline(prompt=prompt,
							   token_indices=list(token_indices),
							   guidance_scale=7.5,
							   generator=generator,
							   num_inference_steps=50,
							   max_iter_to_alter=25,
							   ).images
		return images


# https://huggingface.co/docs/diffusers/main/api/pipelines/deepfloyd_if#text-to-image-generation
class DeepFloyd(AbstractModel):
	def __init__(self, ckpt: dict = {"pipeline": ["DeepFloyd/IF-I-XL-v1.0", "DeepFloyd/IF-II-L-v1.0", "stabilityai/stable-diffusion-x4-upscaler"]}, precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		assert len(ckpt["pipeline"]) == 3
		from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline

		self.pipeline = IFPipeline.from_pretrained(ckpt["pipeline"][0], variant="fp16", torch_dtype=precision).to(device)
		self.super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
			ckpt["pipeline"][1], text_encoder=None, variant="fp16", torch_dtype=torch.float16
		)
		safety_modules = {
			"feature_extractor": self.pipeline.feature_extractor,
			"safety_checker"   : self.pipeline.safety_checker,
			"watermarker"      : self.pipeline.watermarker,
		}
		self.super_res_2_pipe = DiffusionPipeline.from_pretrained(
			"stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
		).to(device)
		self.pipeline.enable_model_cpu_offload()
		self.super_res_1_pipe.enable_model_cpu_offload()
		self.super_res_2_pipe.enable_model_cpu_offload()

	def text2img(self, prompt):
		prompt_embeds, negative_embeds = self.pipeline.encode_prompt(prompt)
		image = self.pipeline(prompt_embeds=prompt_embeds, negative_embeds=negative_embeds, output_type="pt").images
		image = self.super_res_1_pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images
		image = self.super_res_2_pipe(
			prompt=prompt,
			image=image,
		).images
		return image


class KandinskyV21(AbstractModel):
	def __init__(self, ckpt: dict = {"prior-pipeline": "kandinsky-community/kandinsky-2-1-prior", "pipeline": 'kandinsky-community/kandinsky-2-1'}, precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		assert len(ckpt) == 2 and isinstance(ckpt, dict) and "prior-pipeline" in ckpt and "pipeline" in ckpt
		from diffusers import KandinskyPipeline, KandinskyPriorPipeline
		self.pipeline = KandinskyPipeline.from_pretrained(ckpt["pipeline"], torch_dtype=precision).to(device)
		self.pipe_prior = KandinskyPriorPipeline.from_pretrained(ckpt["prior-pipeline"], torch_dtype=precision).to(device)

	def text2img(self, prompt):
		out = self.pipe_prior(prompt)
		image_emb = out.image_embeds
		negative_image_emb = out.negative_image_embeds

		return self.pipeline(prompt, image_embeds=image_emb,
							 negative_image_embeds=negative_image_emb,
							 height=768,
							 width=768,
							 num_inference_steps=100,
							 ).images


class KandinskyV22(AbstractModel):
	def __init__(self, ckpt: dict = {"prior-pipeline": "kandinsky-community/kandinsky-2-2-priorr", "pipeline": 'kandinsky-community/kandinsky-2-2-decoder'}, precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		assert len(ckpt) == 2 and isinstance(ckpt, dict) and "prior-pipeline" in ckpt and "pipeline" in ckpt

		from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
		self.pipeline = KandinskyV22Pipeline.from_pretrained(ckpt["pipeline"], torch_dtype=precision).to(device)
		self.pipe_prior = KandinskyV22PriorPipeline.from_pretrained(ckpt["prior-pipeline"], torch_dtype=precision).to(device)

	def text2img(self, prompt):
		out = self.pipe_prior(prompt)
		image_emb = out.image_embeds
		negative_image_emb = out.negative_image_embeds
		return self.pipeline(
			image_embeds=image_emb,
			negative_image_embeds=negative_image_emb,
			height=768,
			width=768,
			num_inference_steps=50,
		).images


class KandinskyV3(AbstractModel):
	def __init__(self, ckpt: str = 'kandinsky-community/kandinsky-3', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import AutoPipelineForText2Image
		self.pipeline = AutoPipelineForText2Image.from_pretrained(ckpt, torch_dtype=precision, variant="fp16").to(device)
		self.pipeline.enable_model_cpu_offload()

	def text2img(self, prompt):
		generator = torch.Generator(device="cpu").manual_seed(0)
		return self.pipeline(prompt, num_inference_steps=25, generator=generator).images[0]


class Playgroundv2(AbstractModel):
	def __init__(self, ckpt: str = 'playgroundai/playground-v2-1024px-aesthetic', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import DiffusionPipeline
		self.pipeline = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision, add_watermarker=False, vatiant='fp32').to(device)

	def text2img(self, prompt):
		return self.pipeline(
			prompt=prompt,
			guidance_scale=3.0,
		).images


class SelfAttentionGuidance(AbstractModel):
	def __init__(self, ckpt: str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import StableDiffusionSAGPipeline
		self.pipeline = StableDiffusionSAGPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)

	def text2img(self, prompt):
		return self.pipeline(prompt, sag_scale=0.75).images


# Using reference image to add object in generated image
class GLIGEN(AbstractModel):
	def __init__(self, ckpt: str = 'anhnct/Gligen_Text_Image', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import StableDiffusionGLIGENTextImagePipeline
		self.pipeline = StableDiffusionGLIGENTextImagePipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)

	def text2img(self, prompt, boxes, phrases):
		# prompt and phrases
		assert len(boxes) == len(phrases) and len(boxes) > 0 and len(phrases) > 0
		images = self.pipeline(prompt=prompt,
							   gligen_scheduled_sampling_beta=1,
							   gligen_phrases=phrases,
							   output_type="pil",
							   gligen_boxes=boxes,
							   num_inference_steps=50,
							   ).images
		return images


class LDM3D(AbstractModel):
	def __init__(self, ckpt: str = 'Intel/ldm3d-4c', precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import StableDiffusionLDM3DPipeline
		self.pipeline = StableDiffusionLDM3DPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)

	def text2img(self, prompt):
		output = self.pipeline(prompt)
		return output.rgb


class StableUnClip(AbstractModel):
	def __init__(self, ckpt: dict = {"prior_model": "kakaobrain/karlo-v1-alpha", "prior_text_model": "openai/clip-vit-large-patch14", "pipeline": "stabilityai/stable-diffusion-2-1-unclip-small"}, precision: torch.dtype = torch.float16,
				 device: torch.device = torch.device("cuda")):
		from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline
		from diffusers.models import PriorTransformer
		from transformers import CLIPTokenizer, CLIPTextModelWithProjection
		prior = PriorTransformer.from_pretrained(ckpt["prior_model"], subfolder="prior", torch_dtype=precision)
		prior_tokenizer = CLIPTokenizer.from_pretrained(ckpt['prior_text_model'])
		prior_text_model = CLIPTextModelWithProjection.from_pretrained(ckpt['prior_text_model'], torch_dtype=precision)
		prior_scheduler = UnCLIPScheduler.from_pretrained(ckpt["prior_model"], subfolder="prior_scheduler")
		prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)
		pipe = StableUnCLIPPipeline.from_pretrained(
			ckpt['pipeline'],
			torch_dtype=precision,
			# variant="fp16",
			prior_tokenizer=prior_tokenizer,
			prior_text_encoder=prior_text_model,
			prior=prior,
			prior_scheduler=prior_scheduler,
		)
		self.pipeline = pipe.to(device)

	def text2img(self, prompt):
		images = self.pipeline(prompt).images
		return images


class SDXLLightning(AbstractModel):
	def __init__(self, ckpt: dict = {"repo": "ByteDance/SDXL-Lightning", "base": "stabilityai/stable-diffusion-xl-base-1.0", "step": 4, "_ckpt": "dxl_lightning_4step_unet.safetensors", "use_unet": True}, precision: torch.dtype = torch.float16,
				 device: torch.device = torch.device("cuda")):
		assert len(ckpt) == 5 and isinstance(ckpt, dict) and "repo" in ckpt and "base" in ckpt and "_ckpt" in ckpt and "step" in ckpt and "use_unet" in ckpt
		from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
		from huggingface_hub import hf_hub_download
		from safetensors.torch import load_file
		device_name = "cuda" if device.type == "cuda" else "cpu"
		if ckpt["use_unet"]:
			print(torch.cuda.is_available())
			unet = UNet2DConditionModel.from_config(ckpt['base'], subfolder="unet").to("cuda", precision)
			unet.load_state_dict(load_file(hf_hub_download(ckpt['repo'], ckpt['_ckpt']), "cuda"))
			self.pipeline = StableDiffusionXLPipeline.from_pretrained(ckpt['base'], unet=unet, torch_dtype=precision, variant="fp16").to(device)
		else:
			self.pipeline = StableDiffusionXLPipeline.from_pretrained(ckpt['base'], torch_dtype=precision, variant="fp16").to(device)
		self.pipeline.scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing="trailing")
		self.step = ckpt["step"]

	def text2img(self, prompt):
		images = self.pipeline(prompt, num_inference_steps=self.step, guidance_scale=0).images
		return images


class LatentConsistencyModel(AbstractModel):
	def __init__(self, ckpt: str = "SimianLuo/LCM_Dreamshaper_v74", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import DiffusionPipeline
		self.pipeline = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)

	def text2img(self, prompt):
		images = self.pipeline(prompt=prompt, num_inference_steps=4, guidance_scale=8.0).images
		return images


class LDM(AbstractModel):
	def __init__(self, ckpt: str = "CompVis/ldm-text2im-large-256", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import DiffusionPipeline
		self.pipeline = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)

	def text2img(self, prompt):
		images = self.pipeline([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images
		return images


class MultiDiffusion(AbstractModel):
	def __init__(self, ckpt: str = "stabilityai/stable-diffusion-2-base", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler
		scheduler = DDIMScheduler.from_pretrained(ckpt, subfolder="scheduler")

		self.pipeline = StableDiffusionPanoramaPipeline.from_pretrained(
			ckpt, scheduler=scheduler, torch_dtype=precision
		).to(device)

	def text2img(self, prompt):
		images = self.pipeline(prompt).images
		return images


class DiT(AbstractModel):
	def __init__(self, ckpt: str = "facebook/DiT-XL-2-256", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		from diffusers import DiTPipeline, DPMSolverMultistepScheduler
		self.pipeline = DiTPipeline.from_pretrained(ckpt, torch_dtype=precision)
		self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
		self.pipeline = self.pipeline.to(device)

	def text2img(self, prompt):
		if isinstance(prompt, list):
			# prompt = prompt.lower().split()
			# lowercase_lables = {k.lower():v for k,v in self.pipeline.labels.items()}
			# prompt = [word for word in prompt if word in lowercase_lables]
			class_ids = self.pipeline.get_label_ids(prompt)
			if len(class_ids) == 0:
				raise ValueError("Pick words that exist in ImageNet!")
		else:
			raise ValueError("Prompt should be a list of words")
		generator = torch.manual_seed(33)
		images = self.pipeline(class_labels=class_ids, num_inference_steps=25, generator=generator).images
		return images


class PixArt(AbstractModel):
	def __init__(self, ckpt: str = "PixArt-alpha/PixArt-XL-2-1024-MS", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
		self.ckpt = ckpt
		self.precision = precision
		self.device = device
		from transformers import T5EncoderModel
		self.text_encoder = T5EncoderModel.from_pretrained(
			ckpt,
			subfolder="text_encoder",
			load_in_8bit=True,
		)

	def text2img(self, prompt):
		from diffusers import PixArtAlphaPipeline

		self.pipeline = PixArtAlphaPipeline.from_pretrained(self.ckpt, text_encoder=self.text_encoder, transformer=None, ).to(self.device)

		prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = self.pipeline.encode_prompt(prompt)
		import gc

		def flush():
			gc.collect()
			torch.cuda.empty_cache()

		del self.text_encoder
		del self.pipeline
		flush()

		pipe = PixArtAlphaPipeline.from_pretrained(
			self.ckpt,
			text_encoder=None,
			torch_dtype=self.precision,
		).to(self.device)

		latents = pipe(
			negative_prompt=None,
			prompt_embeds=prompt_embeds,
			negative_prompt_embeds=negative_embeds,
			prompt_attention_mask=prompt_attention_mask,
			negative_prompt_attention_mask=negative_prompt_attention_mask,
			num_images_per_prompt=1,
			output_type="latent",
		).images

		del pipe.transformer
		flush()
		# image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
		image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
		image = pipe.image_processor.postprocess(image, output_type="pil")

		# image = pipe.image_processor.postprocess(image, output_type="pil")[0]

		return image
