import json

from flask import Flask, jsonify
from app.service.StockService import get_hello_message
from app.Prediction.main import *
from app.config import db, app
from app.Entity.Stock import Stock


# app = Flask(__name__)

@app.route('/prediction')
def Prediction():
    prediction = btc_prediction()

    # result = Stock.query.filter_by(name="BTC").first()
    # path = result.path
    #
    # with open(path, 'w') as f:
    #     json.dump(prediction, f, indent=2)
    #
    # response = app.response_class(status=200)
    #
    # return response


# @app.route('/<name>')
# def findStockPathByName(name):
#     result = Stock.query.filter_by(name=name).first()
#     path = result.path
#
#     with open(path, 'r') as json_file:
#         data = json.load(json_file)
#
#     return jsonify(data)
