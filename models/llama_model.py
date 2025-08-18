import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaWrapper:
    def __init__(self, model_id="meta-llama/Llama-3.2-1B-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, instruction: str, max_new_tokens=128) -> str:
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
