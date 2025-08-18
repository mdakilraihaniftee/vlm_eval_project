import json
from tqdm import tqdm

class QAEvaluator:
    def __init__(self, model_wrapper, output_path="results_qa.jsonl", max_attempts=1):
        self.model = model_wrapper
        self.output_path = output_path
        self.max_attempts = max_attempts

    def run(self, dataset):
        # clear file at start
        with open(self.output_path, "w") as f:
            pass

        for item in tqdm(dataset, desc="Evaluating QA"):
            question = item["question"]
            reference = item.get("answer", "")  # some samples may not have gold answer

            response = self.model.generate(question)

            result = {
                "id": item.get("id", None),
                "question": question,
                "reference": reference,
                "response": response,
            }

            # ✅ Print live result
            print(f"\n--- ID: {item.get('id', 'NA')} ---")
            print(f"Q: {question}")
            print(f"A(model): {response}")
            print(f"A(ref): {reference}\n")

            # Save to file
            with open(self.output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
