import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.constant import TARGET_COLUMN, MODEL_FILE_NAME, MODEL_FILE_EXTENSION
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class PredictionFileDetail:
    prediction_file_path: str
    prediction_file_name: str


class PredictionPipeline:
    """
    Handles prediction for uploaded CSV files.
    """

    def __init__(self, request) -> None:
        self.request = request
        self.utils = MainUtils()

    def _get_latest_artifact_dir(self) -> str:
        """
        Find the most recent artifacts subdirectory.
        """
        try:
            artifacts_root = "artifacts"
            if not os.path.isdir(artifacts_root):
                raise Exception("No artifacts directory found. Please run training first.")

            subdirs = [
                os.path.join(artifacts_root, d)
                for d in os.listdir(artifacts_root)
                if os.path.isdir(os.path.join(artifacts_root, d))
            ]
            if not subdirs:
                raise Exception("No artifact runs found. Please run training first.")

            latest_dir = max(subdirs, key=os.path.getmtime)
            logging.info(f"Using latest artifact directory for prediction: {latest_dir}")
            return latest_dir
        except Exception as e:
            raise CustomException(e, sys)

    def _load_preprocessor(self):
        try:
            latest_artifact_dir = self._get_latest_artifact_dir()
            preprocessor_path = os.path.join(
                latest_artifact_dir, "data_transformation", "preprocessing.pkl"
            )
            logging.info(f"Loading preprocessor from: {preprocessor_path}")
            return self.utils.load_object(preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def _load_model(self):
        try:
            latest_artifact_dir = self._get_latest_artifact_dir()
            model_file_path = os.path.join(
                latest_artifact_dir, "model", f"{MODEL_FILE_NAME}{MODEL_FILE_EXTENSION}"
            )
            logging.info(f"Loading model from: {model_file_path}")
            return self.utils.load_object(model_file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self) -> PredictionFileDetail:
        """
        Execute prediction on uploaded CSV and return details of the prediction file.
        """
        logging.info("Starting PredictionPipeline.")

        try:
            # Get uploaded file from request
            file_storage = self.request.files.get("file")
            if file_storage is None or file_storage.filename == "":
                raise Exception("No file provided for prediction.")

            latest_artifact_dir = self._get_latest_artifact_dir()
            prediction_dir = os.path.join(latest_artifact_dir, "prediction")
            os.makedirs(prediction_dir, exist_ok=True)

            input_file_path = os.path.join(prediction_dir, file_storage.filename)
            file_storage.save(input_file_path)
            logging.info(f"Uploaded file saved at: {input_file_path}")

            # Read and preprocess data
            df = pd.read_csv(input_file_path)
            df = self.utils.remove_unwanted_spaces(df)
            df.replace("?", np.nan, inplace=True)

            if TARGET_COLUMN in df.columns:
                features_df = df.drop(columns=[TARGET_COLUMN])
            else:
                features_df = df.copy()

            preprocessor = self._load_preprocessor()
            model = self._load_model()

            transformed_features = preprocessor.transform(features_df)
            predictions = model.predict(transformed_features)

            result_df = df.copy()
            result_df["prediction"] = predictions

            prediction_file_name = f"prediction_{Path(file_storage.filename).stem}.csv"
            prediction_file_path = os.path.join(prediction_dir, prediction_file_name)

            result_df.to_csv(prediction_file_path, index=False)
            logging.info(f"Prediction file saved at: {prediction_file_path}")

            return PredictionFileDetail(
                prediction_file_path=prediction_file_path,
                prediction_file_name=prediction_file_name,
            )

        except Exception as e:
            raise CustomException(e, sys)

