from pathlib import Path
import pandas as pd
import joblib
from PIL import Image

# LOAD CSV
def load_csv(file_path: Path):
    """
    Loads a CSV file if it exists.
    Returns a pandas DataFrame or None.
    """
    if file_path.exists():
        return pd.read_csv(file_path)

    return None
# LOAD MODEL

def load_model(file_path: Path):
    """
    Loads a serialized model.
    """
    if file_path.exists():
        return joblib.load(file_path)

    return None
# LOAD IMAGE

def load_image(file_path: Path):
    """
    Loads an image using PIL.
    """
    if file_path.exists():
        return Image.open(file_path)

    return None
# FILE EXISTS
def exists(file_path: Path):
    return file_path.exists()