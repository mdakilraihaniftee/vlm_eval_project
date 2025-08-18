

```bash
vlm_eval/
│
├─ dataset/
│  ├─ __init__.py
│  └─ hades_dataset.py           # Load and preprocess HADES dataset
│
├─ models/
│  ├─ __init__.py                # Imports all model wrappers
│  ├─ llava_model.py             # LLaVA wrapper
│  ├─ qwen_model.py              # Qwen wrapper
│  └─ openai_model.py            # OpenAI GPT-4o wrapper
│
├─ evaluation/
│  ├─ __init__.py
│  └─ evaluator.py               # Core evaluator class, prints and saves results
│
├─ run_eval.py                   # Main Python script to run evaluation
├─ requirements.txt              # Required packages
└─ .env                          # OpenAI API key (for GPT-4o)
```


Setup

Clone the repo:

```bash

git clone https://github.com/mdakilraihaniftee/vlm_eval_project
cd vlm_eval_project

```



Install dependencies:
```bash
pip install -r requirements.txt
```

Create a .env file for OpenAI API (if using GPT-4o):
```bash
OPENAI_API_KEY=your_api_key_here
```
Usage

Run evaluation with any model:

LLaVA
```bash
python run_eval.py --model llava --output llava_results.jsonl
```

Qwen 3B
```bash
python run_eval.py --model qwen3b --output qwen3b_results.jsonl
```

Qwen 7B
```bash
python run_eval.py --model qwen7b --output qwen7b_results.jsonl
```

OpenAI GPT-4o-mini
```bash
python run_eval.py --model openai --openai_model_id gpt-4o-mini --output openai_results.jsonl
```

Optional arguments:

--step last or --step all → evaluate only last step images or all images.

--output → path to JSONL file for saving results.

How it Works

Load HADES dataset (dataset/hades_dataset.py).

Initialize chosen model (models/).

Evaluator (evaluation/evaluator.py) generates a response for each image + instruction.

Prints each step output in console.

Saves all results in JSONL format with:

{
    "id": "1234",
    "scenario": "...",
    "keywords": "...",
    "step": 5,
    "category": "...",
    "instruction": "...",
    "response": "Generated output here"
}

Notes

OpenAI API calls are one image per request.

LLaVA/Qwen run locally using GPU (CUDA recommended).

Can extend to any new VLM by creating a wrapper under models/.