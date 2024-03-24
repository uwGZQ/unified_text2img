# Unified Text-to-Image Generation

This repository provides a unified interface for generating images from text prompts using various state-of-the-art models including DALL-E, Stable Diffusion, and many others. It's designed to simplify the process of integrating and switching between different text-to-image models.

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
from text2img import Text2Img

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

When adding a new model, ensure to include any additional dependencies in `requirements.txt` and make a pull request.

## Supported Models

The library currently supports a wide range of models, including but not limited to:

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

## Extending the Library

To add a new model:

1. Implement a new class for the model inheriting from `AbstractModel`.
2. Define the `text2img` method according to the model's requirements.
3. Update the `text2img_model` dictionary with the new model's details.
4. Ensure to add any model-specific dependencies to `requirements.txt`.

## Contributing

Contributions are welcome! If you add a new model or make improvements, please submit a pull request.

## Issues & Support

For bugs, issues, or feature requests, please file an issue through the GitHub issue tracker.

