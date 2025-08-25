import json
from tqdm import tqdm
from utils.asr import Judge

class ScoreEvaluation:
	def __init__(self, input_jsonl, output_jsonl):
		self.input_jsonl = input_jsonl
		self.output_jsonl = output_jsonl
		self.judge = Judge()

	def run(self, dataset=None):
		with open(self.input_jsonl, 'r', encoding='utf-8') as f:
			data = [json.loads(line) for line in f]

		# Build a lookup for dataset prompts by id if dataset is provided
		dataset_lookup = {item.get('id'): item for item in dataset} if dataset else {}

		output = []
		for item in tqdm(data, desc='Scoring'):
			entry_id = item.get('id')
			instruction = item.get('instruction')
			response = item.get('response')
			# If dataset has more info for this id, use its question as prompt
			if dataset and entry_id in dataset_lookup:
				instruction = dataset_lookup[entry_id].get('question', instruction)
			# Use Judge scoring logic
			score_result = self.judge.openai_text_wrapper(instruction, response)
			merged = {
				'id': entry_id,
				'instruction': instruction,
				'response': response,
				'score': score_result.get('score'),
				'Toxicity': score_result.get('Toxicity')
			}
			output.append(merged)

		with open(self.output_jsonl, 'w', encoding='utf-8') as f:
			for entry in output:
				f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Example usage:
# scorer = ScoreEvaluation('llava_results.jsonl', 'llava_asr_score_results.jsonl')
# scorer.run()
      