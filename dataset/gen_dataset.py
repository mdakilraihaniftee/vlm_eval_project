from datasets import load_dataset

def load_gen_dataset(limit=None):
    """
    Load text prompts for generation from Maxx0/sexting-nsfw-adultconten.
    """
    ds = load_dataset("Maxx0/sexting-nsfw-adultconten")["train"]  # dataset only has 'train'

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return ds
