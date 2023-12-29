import json

from flask import Flask, jsonify
from app.service.StockService import get_hello_message
from app.Prediction.main import prediction
from app.config import db, app
from app.Entity.Stock import Stock

# app = Flask(__name__)

@app.route('/prediction/BTC')
def stockPrediction():
    BTC_prediction = prediction()

    result = Stock.query.filter_by(name="BTC").first()
    path = result.path

    with open(path, 'w') as f:
        json.dump(BTC_prediction, f, indent=2)

    response = app.response_class(status=200)

    return response
