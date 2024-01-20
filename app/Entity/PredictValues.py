from app.config import db


class PredictValues(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    coin_predict_id = db.Column(db.Integer())
    step = db.Column(db.Integer())
    value = db.Column(db.Float())

    def __init__(self, coin_predict_id, value, step):
        self.coin_predict_id = coin_predict_id
        self.value = value
        self.step = step
