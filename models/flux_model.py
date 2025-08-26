import torch
from diffusers import FluxPipeline
from dotenv import load_dotenv
import os

# Load HF token from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")


class FluxWrapper:
    def __init__(self, model_id="black-forest-labs/FLUX.1-Krea-dev", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize pipeline
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            use_auth_token=hf_token  # ensures private model access
        )

        # VRAM optimization
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def generate(self, prompt: str, height=1024, width=1024, guidance_scale=4.5):
        image = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale
        ).images[0]
        return image
