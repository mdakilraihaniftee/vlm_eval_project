import argparse
from dataset import load_gen_dataset
from evaluation import GenEvaluator
from models import StableDiffusionWrapper, FluxWrapper  # import both wrappers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="gen_outputs")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["sd", "flux"],
        default="sd",
        help="Choose which model to use: 'sd' for Stable Diffusion, 'flux' for Flux"
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_gen_dataset(limit=args.limit)

    # Pick model wrapper
    if args.model_type == "sd":
        model = StableDiffusionWrapper("stable-diffusion-v1-5/stable-diffusion-v1-5")
    elif args.model_type == "flux":
        model = FluxWrapper("black-forest-labs/FLUX.1-Krea-dev")

    # Run evaluation
    evaluator = GenEvaluator(model_wrapper=model, output_dir=args.output_dir)
    evaluator.run(dataset)
