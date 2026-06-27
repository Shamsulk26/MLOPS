import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        HF_TOKEN = None

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found.")

api = HfApi(token=HF_TOKEN)
repo_id = "Shamsul26/Wellness-Tourism-App"  # Using a dedicated space name avoids 404/conflict
repo_type = "space"

# Ensure the space repository exists on Hugging Face before committing
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' exists.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="streamlit", private=False)

# Upload your deployment app files
api.upload_folder(
    folder_path="mls/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
)
print("Application folder successfully uploaded!")
