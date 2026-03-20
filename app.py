import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the folder where app.py lives (project root), so it works no matter where you run from
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

from flask import Flask, render_template, jsonify, request, send_file, redirect, url_for
from src.exception import CustomException
from src.logger import logging as lg

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    # Redirect to prediction page as the main UI
    return redirect(url_for("predict"))


@app.route("/train")
def train_route():
    try:
        if not os.getenv("MONGO_DB_URL"):
            return (
                "MONGO_DB_URL is not set. Create a .env file in the project root with:\n"
                "MONGO_DB_URL=mongodb+srv://your-connection-string",
                503,
            )
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return "Training Completed."
    except Exception as e:
        raise CustomException(e, sys)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    
    try:
        if request.method == 'POST':
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipeline()

            lg.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_file_name,
                            as_attachment= True)
        
        else:
            return render_template('prediction.html')

    except Exception as e:
        raise CustomException(e,sys)
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug= True)