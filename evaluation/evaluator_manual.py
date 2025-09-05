import os
import json
from tqdm import tqdm
from PIL import Image

class Evaluator_Manual:
    def __init__(self, model_wrapper, output_path="results.jsonl"):
        self.model = model_wrapper
        self.output_path = output_path

    def run(self, images_folder, instruction):
        # Clear the results file at the start
        with open(self.output_path, "w") as f:
            pass  

        # Collect all image files
        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        for step, filename in enumerate(tqdm(image_files, desc="Evaluating")):
            image_path = os.path.join(images_folder, filename)
            image = Image.open(image_path).convert("RGB")

            # Generate response from model
            response = self.model.generate(image, instruction)

            result = {
                "id": filename,
                "step": step,
                "instruction": instruction,
                "response": response,
            }

            # ✅ Print live output
            print(f"\n--- Step {step} | File: {filename} ---")
            print(f"Instruction: {instruction}")
            print(f"Response: {response}\n")

            # Save result
            with open(self.output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
