from datasets import load_dataset

def load_gen_dataset(dataset_name="Maxx0/sexting-nsfw-adultconten", split="train", limit=None):
    """
    Load dataset from HuggingFace Hub by link.

    Args:
        dataset_name (str): HuggingFace dataset path, e.g. "Maxx0/sexting-nsfw-adultconten" or "AIML-TUDA/i2p"
        split (str): which split to load (default: "train")
        limit (int, optional): number of samples to select

    Returns:
        Dataset: HuggingFace Dataset object
    """
    ds = load_dataset(dataset_name)[split]

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return ds
