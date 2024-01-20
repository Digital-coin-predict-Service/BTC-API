from app.config import db, app
from app.Entity.CoinPredict import CoinPredict
from sqlalchemy import desc

class CoinPredictRepository:

    def saveCoinPredict(self, coinId):
        db.session.add(CoinPredict(coin_id=coinId))

    def findByCoinId(self, stockId):
        return CoinPredict.query.order_by(desc(CoinPredict.predict_at)).filter_by(coin_id=stockId).first()