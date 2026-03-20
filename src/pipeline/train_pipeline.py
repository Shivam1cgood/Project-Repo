import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.constant import artifact_folder, MODEL_FILE_NAME, MODEL_FILE_EXTENSION
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation


@dataclass
class TrainingPipelineConfig:
    model_dir: str = os.path.join(artifact_folder, "model")
    model_file_path: str = os.path.join(
        model_dir, f"{MODEL_FILE_NAME}{MODEL_FILE_EXTENSION}"
    )
    model_config_path: str = os.path.join("config", "model.yaml")


class TrainingPipeline:
    """
    Orchestrates data ingestion, validation, transformation and model training.
    """

    def __init__(self) -> None:
        self.config = TrainingPipelineConfig()
        self.utils = MainUtils()

    def _load_model_config(self) -> dict:
        try:
            return self.utils.read_yaml_file(self.config.model_config_path)
        except Exception as e:
            raise CustomException(e, sys)

    def _build_models_from_config(self, model_config: dict) -> dict:
        """
        Build estimator instances and param grids from YAML configuration.
        """
        try:
            models_section = model_config.get("model_selection", {}).get("model", {})

            models = {}
            param_grids = {}

            for model_name, model_detail in models_section.items():
                search_param_grid = model_detail.get("search_param_grid", {})

                if model_name == "XGBClassifier":
                    estimator = XGBClassifier(
                        use_label_encoder=False,
                        eval_metric="logloss",
                        n_jobs=-1,
                    )
                elif model_name == "GaussianNB":
                    estimator = GaussianNB()
                elif model_name == "LogisticRegression":
                    estimator = LogisticRegression()
                else:
                    # Skip unknown models
                    continue

                models[model_name] = estimator
                param_grids[model_name] = search_param_grid

            return models, param_grids

        except Exception as e:
            raise CustomException(e, sys)

    def _select_and_train_best_model(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
    ):
        try:
            model_config = self._load_model_config()
            models, param_grids = self._build_models_from_config(model_config)

            if not models:
                raise Exception("No valid models found in model configuration.")

            best_model = None
            best_score = -np.inf
            best_model_name = None

            for name, estimator in models.items():
                logging.info(f"Starting grid search for model: {name}")
                param_grid = param_grids.get(name, {})

                grid_search = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1,
                    verbose=1,
                )
                grid_search.fit(x_train, y_train)

                y_pred = grid_search.best_estimator_.predict(x_test)
                score = accuracy_score(y_test, y_pred)

                logging.info(
                    f"Model: {name}, Best Params: {grid_search.best_params_}, "
                    f"Validation Accuracy: {score}"
                )

                if score > best_score:
                    best_score = score
                    best_model = grid_search.best_estimator_
                    best_model_name = name

            if best_model is None:
                raise Exception("No model could be trained successfully.")

            logging.info(
                f"Best model selected: {best_model_name} with validation accuracy: {best_score}"
            )

            os.makedirs(self.config.model_dir, exist_ok=True)
            self.utils.save_object(self.config.model_file_path, best_model)

            logging.info(f"Saved best model at: {self.config.model_file_path}")

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self) -> None:
        """
        End-to-end training pipeline.
        """
        logging.info("Starting TrainingPipeline.")

        try:
            # Data ingestion
            data_ingestion = DataIngestion()
            raw_data_dir = data_ingestion.initiate_data_ingestion()
            logging.info(f"Raw data exported to directory: {raw_data_dir}")

            # Data validation
            data_validation = DataValidation(raw_data_store_dir=Path(raw_data_dir))
            valid_data_dir = data_validation.initiate_data_validation()
            logging.info(f"Validated data directory: {valid_data_dir}")

            # Data transformation
            data_transformation = DataTransformation(valid_data_dir=valid_data_dir)
            (
                x_train,
                y_train,
                x_test,
                y_test,
                preprocessor_path,
            ) = data_transformation.initiate_data_transformation()

            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            # Model training and selection
            best_model_name, best_score = self._select_and_train_best_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )

            logging.info(
                f"Training pipeline completed. Best model: {best_model_name}, "
                f"Validation accuracy: {best_score}"
            )

        except Exception as e:
            raise CustomException(e, sys)

