import argparse
from dataset import load_gen_dataset
from evaluation import GenEvaluator
from models import StableDiffusionWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="gen_outputs")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    dataset = load_gen_dataset(limit=args.limit)
    model = StableDiffusionWrapper("stable-diffusion-v1-5/stable-diffusion-v1-5")

    evaluator = GenEvaluator(model_wrapper=model, output_dir=args.output_dir)
    evaluator.run(dataset)
