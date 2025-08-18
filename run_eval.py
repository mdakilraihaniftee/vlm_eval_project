import argparse
from dataset import load_hades
from evaluation import Evaluator
from models import LlavaWrapper, QwenVLWrapper, OpenAIWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava", 
                        choices=["llava", "qwen3b", "qwen7b", "openai"])
    parser.add_argument("--output", type=str, default="results.jsonl")
    parser.add_argument("--step", type=str, default="last", choices=["last", "all"])
    parser.add_argument("--openai_model_id", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    dataset = load_hades(step=args.step)

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

    evaluator = Evaluator(model_wrapper=model, output_path=args.output)
    evaluator.run(dataset)
