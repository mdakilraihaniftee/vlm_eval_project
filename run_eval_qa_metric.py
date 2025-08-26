#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from evaluation import ScoreEvaluation_Gemini, ScoreEvaluation
from utils.asr import Judge  # import Judge from project root asr.py


def main():
    ap = argparse.ArgumentParser(description="Score LLaVA results using ASR judge.")
    ap.add_argument("--in", dest="inp", required=True, help="Path to llava_results.jsonl")
    ap.add_argument("--out", dest="outp", required=True, help="Path to output jsonl (scored)")
    ap.add_argument(
        "--backend",
        choices=["openai", "gemini"],
        default="openai",
        help="Which evaluation backend to use (default: openai)",
    )
    args = ap.parse_args()

    judge = Judge()

    if args.backend == "openai":
        evaluator = ScoreEvaluation(args.inp, args.outp, judge)
    else:  # gemini
        evaluator = ScoreEvaluation_Gemini(args.inp, args.outp)

    evaluator.run()


if __name__ == "__main__":
    main()
