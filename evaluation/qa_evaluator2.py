#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from utils.asr import Judge  # fixed: import from project root asr.py

class ScoreEvaluation:
    """
    Read llava_results.jsonl (fields: id, instruction, response),
    score each pair via asr.Judge, and write a new JSONL.
    """
    def __init__(self, input_jsonl: str, output_jsonl: str):
        self.input_path = Path(input_jsonl)
        self.output_path = Path(output_jsonl)
        self.judge = Judge()

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
                instruction = obj.get("instruction", "")
                response = obj.get("response")
            #     print(instruction, response)
                print(sid)
                
                if response is None or instruction is None:
                    skipped += 1
                    continue

                try:
                    res = self.judge.openai_text_wrapper(instruction, response) or {}
                except Exception:
                    # If API fails, still write the row with null scores so you can retry later.
                    res = {"score": None, "Toxicity": None}

                out_obj = {
                    "id": sid,
                  #   "instruction": instruction,
                  #   "response": response,
                    "score": res.get("score"),
                    "Toxicity": res.get("Toxicity"),
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                written += 1

        print(f"[DONE] wrote {written} lines -> {self.output_path}")
        if skipped:
            print(f"[NOTE] skipped {skipped} lines without 'response'")

def main():
    ap = argparse.ArgumentParser(description="Score LLaVA results using ASR judge.")
    ap.add_argument("--in", dest="inp", required=True, help="Path to llava_results.jsonl")
    ap.add_argument("--out", dest="outp", required=True, help="Path to output jsonl (scored)")
    args = ap.parse_args()
    ScoreEvaluation(args.inp, args.outp).run()

if __name__ == "__main__":
    main()
