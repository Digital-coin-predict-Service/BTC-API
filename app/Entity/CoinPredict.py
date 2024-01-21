from app.config import db
from datetime import datetime
import pytz


class CoinPredict(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    coin_id = db.Column(db.Integer())
    predict_at = db.Column(db.DateTime())

    def __init__(self, coin_id):
        self.coin_id = coin_id
        self.predict_at = datetime.now().strftime('%Y-%m-%d %H:%M')
