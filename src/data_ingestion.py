import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass #decorator we can directly define class variable with this
#beacuse of this dataclass we can define train_data_path without an init function
class DataIngestionConfig:

    train_data_path:str= os.path.join('artifact','train.csv')# path for data ingetion component like trained data will be saved in this path as well as the outputs
    test_data_path :str= os.path.join('artifact', 'test.csv')
    raw_data_path :str = os.path.join('artifact', 'raw.csv')

    # above are the inputs to data ingestion config


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # above 3 paths are getting stored inside this calss variable

    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component") 
        try:
            #below line u can change the code to read from sql or mongodb
            df = pd.read_csv("notebook/stud.csv")
            logging.info("read the dataset as dataframe")

            # converted dataset into data paths
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("ingestion of data is completed")

            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    # combinign data ingestion and transformation
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ =data_transformation.initiate_data_transformation(train_data, test_data)
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

    