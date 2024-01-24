from app.config import db, app
from app.Prediction.predictions import *
from app.Entity.Stock import Stock
import pyupbit

class StockRepository:
    def findByStockCode(self, stockCode):
        return Stock.query.filter_by(code=stockCode).first()
