from datasets import load_dataset

def load_redteam(limit=None, split = None):
    """
    Load the RedTeam_2K subset from JailBreakV-28K dataset.
    Optionally limit number of samples.
    """
    RedTeam_2K = load_dataset("JailbreakV-28K/JailBreakV-28k", 'RedTeam_2K')["RedTeam_2K"]

    return RedTeam_2K
