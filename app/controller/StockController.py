from flask_apscheduler import scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask

from app.Prediction.predictions import *
from app.config import db, app
from app.Entity.Stock import Stock
from app.Repository.StockRepository import StockRepository
from app.Repository.CoinPredictRepository import CoinPredictRepository
from app.Repository.PredictValuesRepository import PredictValuesRepository

stock_repository = StockRepository()
coin_predict_repository = CoinPredictRepository()
predict_values_repository = PredictValuesRepository()

price_error = {"BTC": 0,
               "ETH": 0,
               "XRP": 0,
               "ASTR": 0,
               "GMT": 0,
               "POWR": 0,
               "SEI": 0,
               "SOL": 0,
               "STX": 0,
               "T": 0}

coin_list = ['BTC', 'ETH', 'XRP', 'ASTR', 'GMT', 'POWR', 'SEI', 'SOL', 'STX', 'T']

#this is function which should be run every minute.
def prediction():
    for coin in coin_list:
        stock = stock_repository.findByStockCode(coin)

        current_price = pyupbit.get_current_price("KRW-" + coin)
        stock.latest_price = abs(price_error[coin] - current_price)
        stock.error_rate = (stock.latest_price / current_price) * 100

        forecast = coin_prediction(coin)
        price_error[coin] = forecast[-10]

        coin_predict_repository.saveCoinPredict(stock.id)
        coin_predict = coin_predict_repository.findByCoinId(stock.id)

        for i, value in enumerate(forecast):
            predict_values_repository.savePredictValue(coin_predict.id, value, i)

    db.session.commit()

    response = app.response_class(status=200)
    return response
