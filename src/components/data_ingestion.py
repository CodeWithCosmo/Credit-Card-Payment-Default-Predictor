import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.logger import logging as lg
from src.utils import read_mongo


@dataclass
class DataIngestionConfig:
    raw_dataset_path: str = os.path.join('artifacts','raw.csv')
    train_dataset_path: str = os.path.join('artifacts','train.csv')
    test_dataset_path: str = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        try:
            self.ingestion_config = DataIngestionConfig()
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self):
        lg.info('Initiating data ingestion')
        try:
            lg.info('Downloading data from MongoDB Cloud')
            data = pd.DataFrame(read_mongo())
            lg.info('Dowloading successful')
            data = data.drop(['_id'],axis=1)
            
            lg.info('Data ingestion completed')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_dataset_path), exist_ok=True)
            lg.info('Feature Selection initiated')

            data.to_csv(self.ingestion_config.raw_dataset_path, index=False, header=True)
            lg.info('Train test split initiated')
            train_set,test_set = train_test_split(data,test_size=0.25,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_dataset_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_dataset_path, index=False, header=True)

            lg.info('Train test split completed')

            return (self.ingestion_config.train_dataset_path,self.ingestion_config.test_dataset_path)
        except Exception as e:
            raise CustomException(e, sys)