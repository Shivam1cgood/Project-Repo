import sys
import os
import pandas as pd
from pathlib import Path

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from src.data_access.phishing_data import PhisingData
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(artifact_folder, "data_ingestion")
    # Local fallback CSV (avoids needing MongoDB during training)
    local_csv_path: str = os.path.join("Notebook implementation", "phising.csv")
    # Use a filename that matches the schema sample pattern
    output_file_name: str = "phishing_08012020_120000.csv"


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def _ingest_from_mongo(self, raw_batch_files_path: str) -> None:
        """
        Primary ingestion path: read all collections from MongoDB and
        save each collection as a CSV under data_ingestion.
        """
        try:
            logging.info("Trying to export data from MongoDB")

            income_data = PhisingData(database_name=MONGO_DATABASE_NAME)
            anything_written = False

            for collection_name, dataset in income_data.export_collections_as_dataframe():
                logging.info(f"Shape of collection '{collection_name}': {dataset.shape}")
                feature_store_file_path = os.path.join(raw_batch_files_path, f"{collection_name}.csv")
                dataset.to_csv(feature_store_file_path, index=False)
                anything_written = True
                logging.info(f"Saved Mongo collection '{collection_name}' to {feature_store_file_path}")

            if not anything_written:
                raise Exception("MongoDB returned no collections to ingest.")

        except Exception as e:
            # Wrap and re-raise so the caller can decide about fallback.
            raise CustomException(e, sys)

    def _ingest_from_local_csv(self, raw_batch_files_path: str) -> None:
        """
        Fallback ingestion path: read the local phishing CSV and save it
        as a single batch file under data_ingestion.
        """
        try:
            logging.info("Falling back to local CSV ingestion")

            csv_path = self.data_ingestion_config.local_csv_path
            if not os.path.exists(csv_path):
                raise Exception(
                    f"Local CSV not found at {csv_path}. "
                    f"Make sure 'Notebook implementation/phising.csv' exists."
                )

            dataframe = pd.read_csv(csv_path)
            logging.info(f"Loaded local CSV with shape: {dataframe.shape}")

            feature_store_file_path = os.path.join(
                raw_batch_files_path, self.data_ingestion_config.output_file_name
            )
            dataframe.to_csv(feature_store_file_path, index=False)
            logging.info(f"Saved ingested data to: {feature_store_file_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_raw_data_dir(self) -> None:
        """
        Orchestrates ingestion.
        - If USE_LOCAL_CSV=1, force local CSV ingestion.
        - Otherwise, try MongoDB first and fall back to local CSV on failure.
        """
        try:
            raw_batch_files_path = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(raw_batch_files_path, exist_ok=True)

            use_local_only = os.getenv("USE_LOCAL_CSV", "0") == "1"

            if use_local_only:
                logging.info("USE_LOCAL_CSV=1 → skipping Mongo, using local CSV only")
                self._ingest_from_local_csv(raw_batch_files_path)
                return

            # Try MongoDB first for interview / production use
            try:
                self._ingest_from_mongo(raw_batch_files_path)
            except Exception as mongo_error:
                logging.warning(
                    f"MongoDB ingestion failed ({mongo_error}). "
                    f"Falling back to local CSV."
                )
                self._ingest_from_local_csv(raw_batch_files_path)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> Path:
        """
        Initiates the data ingestion component of the training pipeline.

        Returns the directory containing ingested CSV files.
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            self.export_data_into_raw_data_dir()

            logging.info("Data ingestion completed (MongoDB and/or local CSV)")
            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            return Path(self.data_ingestion_config.data_ingestion_dir)

        except Exception as e:
            raise CustomException(e, sys) from e