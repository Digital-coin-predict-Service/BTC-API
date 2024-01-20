from app.config import db, app
from app.Prediction.predictions import *
from app.Entity.Stock import Stock
import pyupbit

class StockRepository:
    def findByStockCode(self, stockName):
        return Stock.query.filter_by(code=stockName).first()