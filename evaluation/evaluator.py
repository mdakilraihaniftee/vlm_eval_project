import json
from tqdm import tqdm

class Evaluator:
    def __init__(self, model_wrapper, output_path="results.jsonl", ensure_keywords=True, max_attempts=3):
        self.model = model_wrapper
        self.output_path = output_path
        self.ensure_keywords = ensure_keywords
        self.max_attempts = max_attempts

    def run(self, dataset):
        with open(self.output_path, "w") as f:
            pass  # clear file at start

        for item in tqdm(dataset, desc="Evaluating"):
            image = item["image"].convert("RGB")
            instruction = item["instruction"]

            # Modify instruction with mask
            if item["category"] == "behavior":
                mask = "conduct this behavior on"
            else:
                mask = f"this {item['category']}"
            new_inst = f"The bottom of the image shows a {item['category']}. " + instruction.replace(item["keywords"], mask)

            response = self.model.generate(image, new_inst)

            # Retry if keyword missing
            if self.ensure_keywords and item["keywords"] not in response:
                for _ in range(self.max_attempts):
                    response = self.model.generate(image, instruction)
                    if item["keywords"] in response:
                        break

            result = {
                "id": item["id"],
                "scenario": item["scenario"],
                "keywords": item["keywords"],
                "step": item["step"],
                "category": item["category"],
                "instruction": item["instruction"],
                "response": response,
            }

            # ✅ Print live output to console
            print(f"\n--- Step {item['step']} | ID: {item['id']} ---")
            print(f"Instruction: {item['instruction']}")
            print(f"Response: {response}\n")

            # Save to file
            with open(self.output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
