from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
import os
hf_token = os.getenv("HF_TOKEN")

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

prompt = "A frog holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
).images[0]
image.save("flux-krea-dev.png")
import matplotlib.pyplot as plt
# Plot inline
plt.imshow(image)
plt.axis("off")
plt.show()
