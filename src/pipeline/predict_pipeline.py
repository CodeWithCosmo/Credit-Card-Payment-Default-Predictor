import os,sys
import pandas as pd
from src.logger import logging as lg

from src.exception import CustomException
from flask import request
from src.utils import load_object

from dataclasses import dataclass
        
        
@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str = "prediction_artifacts"
    prediction_file_name:str =  "prediction.csv"
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)



class PredictionPipeline:
    def __init__(self, request: request):

        self.request = request
        self.prediction_file_detail = PredictionFileDetail()



    def save_input_files(self)-> str:
        try:
            pred_file_input_dir = "testing_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)            
            
            input_csv_file.save(pred_file_path)

            return pred_file_path
        
        except Exception as e:
            raise CustomException(e,sys)

    def predict(self,features):
        try:
            lg.info('Initiating prediction')
            model_path = 'artifacts/model.pkl'
            model = load_object(model_path)

            prediction = model.predict(features)

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):
        try:

            prediction_column_name : str = "class"
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'NO', 1:'YES'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            
            os.makedirs( self.prediction_file_detail.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_file_detail.prediction_file_path, index= False)
            lg.info("predictions completed. ")

        except Exception as e:
            raise CustomException(e, sys)       

       
    def initiate_predict_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_file_detail


        except Exception as e:
            raise CustomException(e,sys)         