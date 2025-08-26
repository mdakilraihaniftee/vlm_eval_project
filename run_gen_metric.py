#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from evaluation import ImageScoreEvaluation
from utils import ImageJudge  # <-- make sure your ImageJudge class is here



def main():
    ap = argparse.ArgumentParser(description="Score image results using ImageJudge.")
    ap.add_argument("--in", dest="inp", required=True, help="Path to input JSONL file with image paths")
    ap.add_argument("--out", dest="outp", required=True, help="Path to output JSONL (scored)")
    args = ap.parse_args()
    judge = ImageJudge()
    iscore = ImageScoreEvaluation(args.inp, args.outp, judge)
    iscore.run()


if __name__ == "__main__":
    main()