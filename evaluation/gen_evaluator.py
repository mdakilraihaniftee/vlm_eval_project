import os
from tqdm import tqdm

class GenEvaluator:
    def __init__(self, model_wrapper, output_dir="gen_outputs"):
        self.model = model_wrapper
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, dataset):
        for idx, item in enumerate(tqdm(dataset, desc="Generating")):
            prompt = item["He"]  # dataset column is "text"

            image = self.model.generate(prompt)

            save_path = os.path.join(self.output_dir, f"gen_{idx}.png")
            image.save(save_path)

            # ✅ Print live output
            print(f"\n--- Prompt {idx} ---")
            print(f"Prompt: {prompt}")
            print(f"Saved: {save_path}\n")
