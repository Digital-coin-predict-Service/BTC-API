from app.config import db


class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    code = db.Column(db.String())
    latest_price = db.Column(db.Float())
    error_rate = db.Column(db.Float())
    current_price = db.Column(db.Float())
    estimated_price = db.Column(db.Float())