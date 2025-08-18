import torch
from diffusers import StableDiffusionPipeline

class StableDiffusionWrapper:
    def __init__(self, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to(self.device)

    def generate(self, prompt: str, num_inference_steps=25, guidance_scale=7.5):
        image = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        return image
