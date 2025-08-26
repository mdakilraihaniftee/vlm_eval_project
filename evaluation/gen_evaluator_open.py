#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from utils.asr_image import ImageJudge  # <-- make sure your ImageJudge class is here

class ImageScoreEvaluation:
    """
    Read image_results.jsonl (fields: id, image_path),
    evaluate each image via ImageJudge, and write a new JSONL
    with id, score, and Toxicity.
    """
    def __init__(self, input_jsonl: str, output_jsonl: str):
        self.input_path = Path(input_jsonl)
        self.output_path = Path(output_jsonl)
        self.judge = ImageJudge()

    def _stream_jsonl(self, path: Path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        written, skipped = 0, 0

        with self.output_path.open("w", encoding="utf-8") as fout:
            for obj in self._stream_jsonl(self.input_path):
                sid = obj.get("id")
                image_path = obj.get("image_path")
                print(sid)
                print(image_path)
                if not image_path:
                    skipped += 1
                    continue

                try:
                    res = self.judge.evaluate_image(image_path) or {}
                except Exception:
                    res = {"score": None, "Toxicity": None}

                out_obj = {
                    "id": sid,
                    "image_path": image_path,
                    "score": res.get("score"),
                    "Toxicity": res.get("Toxicity"),
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                written += 1

        print(f"[DONE] wrote {written} lines -> {self.output_path}")
        if skipped:
            print(f"[NOTE] skipped {skipped} lines without 'image_path'")


def main():
    ap = argparse.ArgumentParser(description="Score image results using ImageJudge.")
    ap.add_argument("--in", dest="inp", required=True, help="Path to input JSONL file with image paths")
    ap.add_argument("--out", dest="outp", required=True, help="Path to output JSONL (scored)")
    args = ap.parse_args()
    ImageScoreEvaluation(args.inp, args.outp).run()


if __name__ == "__main__":
    main()
