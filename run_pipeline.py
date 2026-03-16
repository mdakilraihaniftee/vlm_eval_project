import argparse
from dataset import load_redteam
from evaluation import UnifiedEvaluator
from models import LlamaWrapper, QwenWrapper, GemmaWrapper, OpenAITextWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full VLM evaluation pipeline (Gen + Judge serially).")
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "qwen", "gemma", "openai"], help="VLM model to evaluate.")
    parser.add_argument("--judge", type=str, default="gemma", choices=["openai", "gemini", "gemma"], help="Judge model to use.")
    parser.add_argument("--output", type=str, default="unified_results.jsonl", help="Output JSONL file.")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of samples (None = all).")
    parser.add_argument("--model_id", type=str, default=None, help="Custom model ID for the VLM (optional).")
    args = parser.parse_args()

    # Load dataset
    dataset = load_redteam(limit=args.limit)

    # Initialize VLM model
    if args.model == "llama":
        model_id = args.model_id or "meta-llama/Llama-3.2-1B"
        model = LlamaWrapper(model_id)
    elif args.model == "qwen":
        model_id = args.model_id or "Qwen/Qwen3-8B"
        model = QwenWrapper(model_id)
    elif args.model == "gemma":
        model_id = args.model_id or "google/gemma-3-1b-it"
        model = GemmaWrapper(model_id)
    elif args.model == "openai":
        model_id = args.model_id or "gpt-4o-mini"
        model = OpenAITextWrapper(model_id)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Run Unified Evaluator
    evaluator = UnifiedEvaluator(
        model_wrapper=model,
        judge_backend=args.judge,
        output_path=args.output,
        ensure_keywords=True
    )
    
    evaluator.run(dataset)
