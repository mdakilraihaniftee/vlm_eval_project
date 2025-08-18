import argparse
from dataset import load_redteam
from evaluation import QAEvaluator
from models import LlamaWrapper, QwenWrapper, GemmaWrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "qwen", "gemma"])
    parser.add_argument("--output", type=str, default="results_qa.jsonl")
    parser.add_argument("--limit", type=int, default=2000, help="Limit number of samples (None = all)")
    args = parser.parse_args()

    dataset = load_redteam(limit=args.limit)

    if args.model == "llama":
        model = LlamaWrapper("meta-llama/Llama-3.2-1B")
    elif args.model == "qwen":
        model = QwenWrapper("Qwen/Qwen3-8B")
    elif args.model == "gemma":
        model = GemmaWrapper("google/gemma-3-1b-it")
    else:
        raise ValueError("Unsupported model")

    evaluator = QAEvaluator(model_wrapper=model, output_path=args.output)
    evaluator.run(dataset)
