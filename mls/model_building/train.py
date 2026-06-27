import joblib
import mlflow
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from huggingface_hub import HfApi

# 1. Load dataset from HF
DATASET_PATH = "hf://datasets/Shamsul26/MLOPS/tourism.csv"
tourism = pd.read_csv(DATASET_PATH)

# Clean and match exact preprocessing logic
if "Unnamed: 0" in tourism.columns:
    tourism.drop(columns=["Unnamed: 0"], inplace=True)
if "CustomerID" in tourism.columns:
    tourism.drop(columns=["CustomerID"], inplace=True)
tourism['MaritalStatus'] = tourism['MaritalStatus'].replace(['Divorced', 'Unmarried'], 'Single')

target_col = 'ProdTaken'
numeric_features = tourism.select_dtypes(include=['int64', 'float64']).columns.drop(target_col).tolist()
categorical_features = tourism.select_dtypes(include=['object', 'category']).columns.tolist()

X = tourism.drop(columns=[target_col])
y = tourism[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Pipeline steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
    remainder='passthrough'
)
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, max_depth=3, learning_rate=0.05, n_estimators=100, random_state=42)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Train the final model
print("Training model...")
model_pipeline.fit(Xtrain, ytrain)

# 2. Serialize model locally
model_filename = "best_tourism_model_v1.joblib"
joblib.dump(model_pipeline, model_filename)
print(f"Model saved locally as {model_filename}")

# 3. Upload model artifact to Hugging Face Dataset repository
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        pass

if HF_TOKEN:
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=model_filename,
        path_in_repo=model_filename,
        repo_id="Shamsul26/MLOPS",
        repo_type="dataset",
    )
    print("Model successfully uploaded to Hugging Face dataset repo!")
else:
    print("WARNING: HF_TOKEN not found. Model not uploaded to Hugging Face.")

