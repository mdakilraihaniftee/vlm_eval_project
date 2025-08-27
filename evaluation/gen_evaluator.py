import os
import json
from tqdm import tqdm
from datasets import load_dataset

class GenEvaluator:
    def __init__(self, model_wrapper, dataset_name, prompt_column, 
                 output_dir="gen_outputs", jsonl_path="synthetic_img_path.jsonl"):
        """
        Args:
            model_wrapper: your model with .generate(prompt)
            dataset_name (str): HuggingFace dataset path, e.g. "Maxx0/sexting-nsfw-adultconten"
            prompt_column (str): column name containing text prompts, e.g. "He" or "prompt"
            output_dir (str): folder to save generated images
            jsonl_path (str): path to save JSONL with metadata
        """
        self.model = model_wrapper
        self.dataset_name = dataset_name
        self.prompt_column = prompt_column
        self.output_dir = output_dir
        self.jsonl_path = jsonl_path

        os.makedirs(self.output_dir, exist_ok=True)

        # clear existing jsonl if already exists
        if os.path.exists(self.jsonl_path):
            os.remove(self.jsonl_path)

    def load_dataset(self, split="train", limit=None):
        ds = load_dataset(self.dataset_name)[split]

        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        return ds

    def run(self, dataset, project_root="/teamspace/studios/this_studio/vlm_eval_project"):
        for idx, item in enumerate(tqdm(dataset, desc="Generating")):
            # ✅ use chosen column
            if self.prompt_column not in item:
                raise KeyError(
                    f"Column '{self.prompt_column}' not found in dataset. "
                    f"Available: {list(item.keys())}"
                )

            prompt = item[self.prompt_column]

            # generate image
            image = self.model.generate(prompt)

            # save image
            save_path = os.path.join(self.output_dir, f"gen_{idx}.png")
            image.save(save_path)

            # absolute path
            abs_path = os.path.abspath(save_path)

            # append JSONL
            record = {"id": idx + 1, "image_path": abs_path}
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            # live output
            print(f"\n--- Prompt {idx} ---")
            print(f"Prompt: {prompt}")
            print(f"Saved: {save_path}")
            print(f"JSONL: {record}\n")
