from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    try:
        from google.colab import userdata  # only available in Colab
        HF_TOKEN = userdata.get("HF_TOKEN")
    except ImportError:
        HF_TOKEN = None

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not found. Set the HF_TOKEN environment variable in CI "
        "or provide it from Colab userdata when running in Colab."
    )
repo_id = "Shamsul26/MLOPS"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="mls/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
