import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

class LlavaWrapper:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate(self, image, instruction: str) -> str:
        if not isinstance(image, Image.Image):
            image = image.convert("RGB")

        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image"}
            ]}
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        caption = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return caption.strip()
