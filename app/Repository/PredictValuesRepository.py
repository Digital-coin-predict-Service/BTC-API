from app.config import db, app
from app.Entity.PredictValues import PredictValues

class PredictValuesRepository:
    def savePredictValue(self, coin_predict_id, value, step):
        db.session.add(PredictValues(coin_predict_id, value, step))
