import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenVLWrapper:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model & processor
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate(self, image, instruction: str, max_new_tokens: int = 128) -> str:
        if not isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Qwen uses multimodal messages with "messages" API
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            }
        ]

        # Apply chat template
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Preprocess inputs (image + text prompt)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

        # Generate
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        # Decode response
        response = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return response.strip()


class QwenWrapper:
    def __init__(self, model_id="Qwen/Qwen3-8B", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate(self, instruction: str, max_new_tokens=128) -> str:
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()