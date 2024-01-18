import json

from flask import Flask, jsonify
from app.Prediction.predictions import *
from app.config import db, app
from app.Entity.Stock import Stock

@app.route('/prediction')
def prediction():
    # =======================<BTC>=============================
    forecast = btc_prediction()

    result = Stock.query.filter_by(name="BTC").first()
    path = result.path

    with open(path, 'w') as f:
        json.dump(forecast, f, indent=2)

    f.close()

    # =======================<ETH>=============================

    forecast = eth_prediction()

    result = Stock.query.filter_by(name="ETH").first()
    path = result.path

    with open(path, 'w') as f:
        json.dump(forecast, f, indent=2)

    f.close()

    # =======================<XRP>=============================

    # forecast = xrp_prediction()
    #
    # result = Stock.query.filter_by(name="XRP").first()
    # path = result.path
    #
    # with open(path, 'w') as f:
    #     json.dump(forecast, f, indent=2)
    #
    # f.close()
    #
    # # =======================<ASTR>=============================
    #
    # forecast = astr_prediction()
    #
    # result = Stock.query.filter_by(name="ASTR").first()
    # path = result.path
    #
    # with open(path, 'w') as f:
    #     json.dump(forecast, f, indent=2)
    #
    # f.close()
    #
    # # =======================<GMT>=============================
    #
    # forecast = gmt_prediction()
    #
    # result = Stock.query.filter_by(name="GMT").first()
    # path = result.path
    #
    # with open(path, 'w') as f:
    #     json.dump(forecast, f, indent=2)
    #
    # f.close()
    #
    # # =======================<POWR>=============================
    #
    # forecast = powr_prediction()
    #
    # result = Stock.query.filter_by(name="POWR").first()
    # path = result.path
    #
    # with open(path, 'w') as f:
    #     json.dump(forecast, f, indent=2)
    #
    # f.close()
    #
    # # =======================<SEI>=============================
    #
    # forecast = sei_prediction()
    #
    # result = Stock.query.filter_by(name="SEI").first()
    # path = result.path
    #
    # with open(path, 'w') as f:
    #     json.dump(forecast, f, indent=2)
    #
    # f.close()
    #
    # # =======================<SOL>=============================
    #
    # forecast = sol_prediction()
    #
    # result = Stock.query.filter_by(name="SOL").first()
    # path = result.path
    #
    # with open(path, 'w') as f:
    #     json.dump(forecast, f, indent=2)
    #
    # f.close()
    #
    # # =======================<STX>=============================
    #
    # forecast = stx_prediction()
    #
    # result = Stock.query.filter_by(name="STX").first()
    # path = result.path
    #
    # with open(path, 'w') as f:
    #     json.dump(forecast, f, indent=2)
    #
    # f.close()
    #
    # # =======================<T>=============================
    #
    # forecast = t_prediction()
    #
    # result = Stock.query.filter_by(name="T").first()
    # path = result.path
    #
    # with open(path, 'w') as f:
    #     json.dump(forecast, f, indent=2)
    #
    # f.close()

    response = app.response_class(status=200)
    return response

