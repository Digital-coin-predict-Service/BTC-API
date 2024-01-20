from app.config import db
from datetime import datetime
import pytz


class CoinPredict(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    coin_id = db.Column(db.Integer())
    predict_at = db.Column(db.DateTime(), default=datetime.now(pytz.timezone('Asia/Seoul')).replace(second=0, microsecond=0))

    def __init__(self, coin_id):
        self.coin_id = coin_id
