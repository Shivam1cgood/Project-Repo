import sys
from typing import List

import numpy as np
import pandas as pd
from pymongo import MongoClient

from src.constant import *
from src.configuration.mongo_db_connection import MongoDBClient
from src.exception import CustomException
import os


class PhisingData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self,
                 database_name: str):
        """
        """
        try:

            self.database_name = database_name
            self.mongo_url = os.getenv("MONGO_DB_URL")

        except Exception as e:
            raise CustomException(e, sys)

    def get_collection_names(self) -> List:
        """
        Return list of collection names from the configured MongoDB database.
        """
        try:
            client = MongoDBClient(database_name=self.database_name)
            collection_names = client.database.list_collection_names()
            return collection_names
        except Exception as e:
            raise CustomException(e, sys)

    def get_collection_data(self, collection_name: str) -> pd.DataFrame:
        """
        Fetch all documents from a given collection as a pandas DataFrame.
        """
        try:
            client = MongoDBClient(database_name=self.database_name)
            collection = client.database[collection_name]
            records = list(collection.find())

            df = pd.DataFrame(records)

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])

            df = df.replace({"na": np.nan})
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def export_collections_as_dataframe(
            self) -> pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            return dd.DataFrame of collection
            """

            collections = self.get_collection_names()

            for collection_name in collections:
                df = self.get_collection_data(collection_name=collection_name)
                yield collection_name, df



        except Exception as e:
            raise CustomException(e, sys)