# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    try:
        from google.colab import userdata  # only available in Colab
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        HF_TOKEN = None

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not found. Set the HF_TOKEN environment variable in CI "
        "or provide it from Colab userdata when running in Colab."
    )

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Shamsul26/MLOPS/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
