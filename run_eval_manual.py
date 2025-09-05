import argparse
import os
from evaluation import Evaluator_Manual
from models import LlavaWrapper, QwenVLWrapper, OpenAIWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava", 
                        choices=["llava", "qwen3b", "qwen7b", "openai"])
    parser.add_argument("--output", type=str, default="results.jsonl")
    parser.add_argument("--openai_model_id", type=str, default="gpt-4o-mini")
    parser.add_argument("--images_folder", type=str, default="images",
                        help="Path to the local images folder")
    parser.add_argument("--instruction", type=str,  default="Describe the main object in this picture.",
                        help="Common instruction to run on all images")
    args = parser.parse_args()

    # pick model
    if args.model == "llava":
        model = LlavaWrapper("llava-hf/llava-1.5-7b-hf")
    elif args.model == "qwen3b":
        model = QwenVLWrapper("Qwen/Qwen2.5-VL-3B-Instruct")
    elif args.model == "qwen7b":
        model = QwenVLWrapper("Qwen/Qwen2.5-VL-7B-Instruct")
    elif args.model == "openai":
        model = OpenAIWrapper(args.openai_model_id)
    else:
        raise ValueError("Unsupported model")

    # run evaluator on local images
    evaluator = Evaluator_Manual(model_wrapper=model, output_path=args.output)
    evaluator.run(images_folder=args.images_folder, instruction=args.instruction)
