import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# --------------------------------------------------------------
# COMPLETE PROJECT STRUCTURE
# --------------------------------------------------------------
list_of_files = [

    # Root
    "requirements.txt",
    "README.md",
    "setup.py",
    "LICENSE",
    ".gitignore",

    # Config
    "config/training_config.yaml",
    "config/model_config.yaml",
    "config/deployment_config.yaml",

    # Docs
    "docs/ROADMAP.md",
    "docs/ARCHITECTURE.md",
    "docs/API_REFERENCE.md",
    "docs/DEPLOYMENT.md",

    # Logs
    "logs/training/.gitkeep",
    "logs/inference/.gitkeep",

    # Data folders
    "data/raw/electronics/.gitkeep",
    "data/raw/fashion/.gitkeep",
    "data/raw/home/.gitkeep",
    "data/processed/train/.gitkeep",
    "data/processed/val/.gitkeep",
    "data/processed/test/.gitkeep",
    "data/metadata/train_labels.csv",
    "data/metadata/val_labels.csv",
    "data/metadata/test_labels.csv",

    # Model Artifacts
    "models/ensemble_model.pth",
    "models/multimodal_model.pth",
    "models/scaler.pkl",
    "models/tokenizer/.gitkeep",

    # Notebooks
    "notebooks/01_EDA.ipynb",
    "notebooks/02_baseline.ipynb",
    "notebooks/03_resnet_features.ipynb",
    "notebooks/04_multimodal.ipynb",
    "notebooks/05_mtl.ipynb",
    "notebooks/06_evaluation.ipynb",

    # Source Code
    "src/__init__.py",
    "src/data_preparation.py",
    "src/baseline_model.py",
    "src/resnet_feature_extractor.py",
    "src/multimodal_fusion.py",
    "src/multi_task_learning.py",
    "src/trainer.py",
    "src/evaluator.py",
    "src/utils.py",

    # Streamlit App
    "streamlit_app/app.py",
    "streamlit_app/model_handler.py",
    "streamlit_app/config.py",
    "streamlit_app/pages/📊_Analysis.py",
    "streamlit_app/pages/📈_Trends.py",
    "streamlit_app/pages/⚙️_Settings.py",

    # Tests
    "tests/__init__.py",
    "tests/test_data_pipeline.py",
    "tests/test_models.py",
    "tests/test_inference.py",
    "tests/test_api.py",
]


# --------------------------------------------------------------
# CREATE FOLDERS + FILES
# --------------------------------------------------------------
for filepath in list_of_files:

    path = Path(filepath)
    directory, filename = os.path.split(path)

    # Create directory if needed
    if directory != "":
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

    # Create file if does not exist
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        with open(path, "w") as f:
            pass
        logging.info(f"Created file: {path}")

    else:
        logging.info(f"File exists: {path}")
logging.info("Project structure creation complete.")
