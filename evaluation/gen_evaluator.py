import os
import json
from tqdm import tqdm

class GenEvaluator:
    def __init__(self, model_wrapper, output_dir="gen_outputs", jsonl_path="synthetic_img_path.jsonl"):
        self.model = model_wrapper
        self.output_dir = output_dir
        self.jsonl_path = jsonl_path

        os.makedirs(self.output_dir, exist_ok=True)

        # clear existing jsonl if already exists
        if os.path.exists(self.jsonl_path):
            os.remove(self.jsonl_path)

    def run(self, dataset, project_root="/teamspace/studios/this_studio/vlm_eval_project"):
        for idx, item in enumerate(tqdm(dataset, desc="Generating")):
            prompt = item["He"]  # dataset column is "text"

            # generate image
            image = self.model.generate(prompt)

            # save image (inside gen_outputs folder)
            save_path = os.path.join(self.output_dir, f"gen_{idx}.png")
            image.save(save_path)

            # absolute path for JSONL
            abs_path = os.path.abspath(save_path)   # will expand to /teamspace/studios/this_studio/vlm_eval_project/gen_outputs/gen_0.png

            # append record to JSONL
            record = {"id": idx + 1, "image_path": abs_path}
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            # ✅ live output
            print(f"\n--- Prompt {idx} ---")
            print(f"Prompt: {prompt}")
            print(f"Saved: {save_path}")
            print(f"JSONL: {record}\n")
