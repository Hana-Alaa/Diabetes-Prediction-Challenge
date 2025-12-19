import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys

# Add project root to path to ensure imports work correctly found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor # Required for unpickling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference():
    logger.info("Starting Inference Pipeline...")

    # Path constants
    MODEL_PATH = "models/final_model_v2.pkl"
    PREPROCESSOR_PATH = "models/preprocessor_v2.pkl"
    SUBMISSION_PATH = "submissions/submission_v2.csv"
    os.makedirs("submissions", exist_ok=True)


    # 1. Load Data
    logger.info("Loading test data...")
    loader = DataLoader() # Defaults to 'Data' directory
    try:
        # We only need test_df and sample_sub (for IDs)
        _, test_df, sample_sub = loader.load_all()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 2. Load Artifacts
    logger.info("Loading model and preprocessor...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        logger.error("Artifacts not found! Please run 'src/model.py' first.")
        return

    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info("Artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        return

    # 3. Preprocess Test Data
    logger.info("Preprocessing test data...")
    try:
        # Transform test data using the fitted preprocessor
        X_test = preprocessor.transform(test_df)
        logger.info(f"Test shape after preprocessing: {X_test.shape}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return

    # 4. Generate Predictions
    logger.info("Generating predictions...")
    try:
        # We want probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            # Class 1 probability
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            logger.warning("Model lacks predict_proba, using hard predictions (not ideal for AUC).")
            y_pred = model.predict(X_test)
        
        logger.info("Predictions generated.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return

    # 5. Create Submission File
    logger.info("Saving submission file...")
    try:
        # Create dataframe ensuring ID alignment
        # We assume sample_sub has correct IDs in order, but nice to be safe
        submission = pd.DataFrame({
            'id': sample_sub['id'],
            'diagnosed_diabetes': y_pred
        })

        submission.to_csv(SUBMISSION_PATH, index=False)
        logger.info(f"✅ Submission saved to {SUBMISSION_PATH}")
        
        # Log stats sanity check
        logger.info(f"Prediction Stats:\n{submission['diagnosed_diabetes'].describe()}")
        
    except Exception as e:
        logger.error(f"Failed to save submission: {e}")

if __name__ == "__main__":
    run_inference()
