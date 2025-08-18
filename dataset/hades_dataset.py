from datasets import load_dataset

def load_hades(split="test", step="last"):
    dataset = load_dataset("Monosail/HADES")[split]

    if step == "last":
        dataset = [item for item in dataset if item['step'] == 5]
    elif step == "all":
        dataset = list(dataset)

    return dataset
