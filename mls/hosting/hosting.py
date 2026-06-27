from huggingface_hub import HfApi
import os

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

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="mls/deployment",     # the local folder containing your files
    repo_id="Shamsul26/MLOPS",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
