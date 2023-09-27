import sys
from flask import Flask, render_template, request, send_file
from src.exception import CustomException
from src.logger import logging as lg
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/train")
def train():
    try:
        score = TrainPipeline().initiate_train_pipeline()

        return render_template("train.html",text = f"Accuracy Score: {round(score,2)}")

    except Exception as e:
        raise CustomException(e,sys)

@app.route('/predict', methods=['POST', 'GET'])
def predict():    
    try:
        if request.method == 'POST':
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.initiate_predict_pipeline()

            lg.info("Prediction completed, Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_file_name,
                            as_attachment= True)
        else:
            return render_template('predict.html')
        
    except Exception as e:
        raise CustomException(e,sys)    