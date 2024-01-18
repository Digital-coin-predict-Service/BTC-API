import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pyupbit

'''preparing models'''

BTC_path = "app/Prediction/BTC_model/model.h5"
BTC_model = tf.keras.models.load_model(BTC_path)
BTC_model.predict(np.zeros((1, 50, 2)))

ETH_path = "app/Prediction/ETH_model/model.h5"
ETH_model = tf.keras.models.load_model(ETH_path)
ETH_model.predict(np.zeros((1, 50, 2)))

# XRP_path = "app/Prediction/XRP_model/model.h5"
# XRP_model = tf.keras.models.load_model(XRP_path)
# XRP_model.predict(np.zeros((1, 50, 2)))
#
# ASTR_path = "app/Prediction/ASTR_model/model.h5"
# ASTR_model = tf.keras.models.load_model(ASTR_path)
# ASTR_model.predict(np.zeros((1, 50, 2)))
#
# GMT_path = "app/Prediction/GMT_model/model.h5"
# GMT_model = tf.keras.models.load_model(GMT_path)
# GMT_model.predict(np.zeros((1, 50, 2)))
#
# POWR_path = "app/Prediction/POWR_model/model.h5"
# POWR_model = tf.keras.models.load_model(POWR_path)
# POWR_model.predict(np.zeros((1, 50, 2)))
#
# SEI_path = "app/Prediction/SEI_model/model.h5"
# SEI_model = tf.keras.models.load_model(SEI_path)
# SEI_model.predict(np.zeros((1, 50, 2)))
#
# SOL_path = "app/Prediction/SOL_model/mode.h5"
# SOL_model = tf.keras.models.load_model(SOL_path)
# SOL_model.predict(np.zeros((1, 50, 2)))
#
# STX_path = "app/Prediction/STX_model/model.h5"
# STX_model = tf.keras.models.load_model(STX_path)
# STX_model.predict(np.zeros((1, 50, 2)))
#
# T_path = "app/Prediction/T_model/model.h5"
# T_model = tf.keras.models.load_model(T_path)
# T_model.predict(np.zeros((1, 50, 2)))

# =============================================================================

def moving_average(data, window_size):
    # 주어진 윈도우 크기에 따라 가중치를 생성
    weights = np.repeat(1.0, window_size) / window_size

    # 이동 평균 계산
    moving_avg = np.convolve(data, weights, 'valid')

    return moving_avg

scaler = MinMaxScaler()


# =======================<BTC>=============================
def btc_prediction():
    x_step = 50  # length of train_x of each time step
    dividing = 10000

    dataframe = pyupbit.get_ohlcv(ticker="KRW-BTC", interval="minute1", count=x_step+9)

    volume = dataframe['volume'].values
    price = dataframe['value'] / dataframe['volume']
    price = price / dividing

    price = moving_average(price, 10)
    volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))

    volume = scaler.fit_transform(volume.reshape(-1, 1))
    price = scaler.fit_transform(price.reshape(-1, 1))

    data = np.concatenate((price, volume), axis=1)

    prediction = BTC_model.predict(np.array([data]))

    prediction = scaler.inverse_transform(prediction)*10000

    result = np.append(scaler.inverse_transform(price)*10000, prediction)

    return result.flatten()


# =======================<ETH>=============================
def eth_prediction():
    x_step = 50  # length of train_x of each time step
    dividing = 10000

    dataframe = pyupbit.get_ohlcv(ticker="KRW-ETH", interval="minute1", count=x_step+9)

    volume = dataframe['volume'].values
    price = dataframe['value'] / dataframe['volume']
    price = price / dividing

    price = moving_average(price, 10)
    volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))

    volume = scaler.fit_transform(volume.reshape(-1, 1))
    price = scaler.fit_transform(price.reshape(-1, 1))

    data = np.concatenate((price, volume), axis=1)

    prediction = ETH_model.predict(np.array([data]))

    prediction = scaler.inverse_transform(prediction)*10000

    result = np.append(scaler.inverse_transform(price)*10000, prediction)

    return result.flatten()
#
#
# # =======================<XRP>=============================
# def xrp_prediction():
#     x_step = 50  # length of train_x of each time step
#     dividing = 10000
#
#     dataframe = pyupbit.get_ohlcv(ticker="KRW-XRP", interval="minute1", count=x_step+9)
#
#     volume = dataframe['volume'].values
#     price = dataframe['value'] / dataframe['volume']
#     price = price / dividing
#
#     price = moving_average(price, 10)
#     volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))
#
#     volume = scaler.fit_transform(volume.reshape(-1, 1))
#     price = scaler.fit_transform(price.reshape(-1, 1))
#
#     data = np.concatenate((price, volume), axis=1)
#
#     prediction = XRP_model.predict(np.array([data]))
#
#     prediction = scaler.inverse_transform(prediction)*10000
#
#     result = np.append(scaler.inverse_transform(price)*10000, prediction)
#
#     return result.flatten()
#
#
# # =======================<ASTR>=============================
# def astr_prediction():
#     x_step = 50  # length of train_x of each time step
#     dividing = 10000
#
#     dataframe = pyupbit.get_ohlcv(ticker="KRW-ASTR", interval="minute1", count=x_step+9)
#
#     volume = dataframe['volume'].values
#     price = dataframe['value'] / dataframe['volume']
#     price = price / dividing
#
#     price = moving_average(price, 10)
#     volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))
#
#     volume = scaler.fit_transform(volume.reshape(-1, 1))
#     price = scaler.fit_transform(price.reshape(-1, 1))
#
#     data = np.concatenate((price, volume), axis=1)
#
#     prediction = ASTR_model.predict(np.array([data]))
#
#     prediction = scaler.inverse_transform(prediction)*10000
#
#     result = np.append(scaler.inverse_transform(price)*10000, prediction)
#
#     return result.flatten()
#
#
# # =======================<GMT>=============================
# def gmt_prediction():
#     x_step = 50  # length of train_x of each time step
#     dividing = 10000
#
#     dataframe = pyupbit.get_ohlcv(ticker="KRW-GMT", interval="minute1", count=x_step+9)
#
#     volume = dataframe['volume'].values
#     price = dataframe['value'] / dataframe['volume']
#     price = price / dividing
#
#     price = moving_average(price, 10)
#     volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))
#
#     volume = scaler.fit_transform(volume.reshape(-1, 1))
#     price = scaler.fit_transform(price.reshape(-1, 1))
#
#     data = np.concatenate((price, volume), axis=1)
#
#     prediction = GMT_model.predict(np.array([data]))
#
#     prediction = scaler.inverse_transform(prediction)*10000
#
#     result = np.append(scaler.inverse_transform(price)*10000, prediction)
#
#     return result.flatten()
#
#
# # =======================<POWR>=============================
# def powr_prediction():
#     x_step = 50  # length of train_x of each time step
#     dividing = 10000
#
#     dataframe = pyupbit.get_ohlcv(ticker="KRW-POWR", interval="minute1", count=x_step+9)
#
#     volume = dataframe['volume'].values
#     price = dataframe['value'] / dataframe['volume']
#     price = price / dividing
#
#     price = moving_average(price, 10)
#     volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))
#
#     volume = scaler.fit_transform(volume.reshape(-1, 1))
#     price = scaler.fit_transform(price.reshape(-1, 1))
#
#     data = np.concatenate((price, volume), axis=1)
#
#     prediction = POWR_model.predict(np.array([data]))
#
#     prediction = scaler.inverse_transform(prediction)*10000
#
#     result = np.append(scaler.inverse_transform(price)*10000, prediction)
#
#     return result.flatten()
#
#
# # =======================<SEI>=============================
# def sei_prediction():
#     x_step = 50  # length of train_x of each time step
#     dividing = 10000
#
#     dataframe = pyupbit.get_ohlcv(ticker="KRW-SEI", interval="minute1", count=x_step+9)
#
#     volume = dataframe['volume'].values
#     price = dataframe['value'] / dataframe['volume']
#     price = price / dividing
#
#     price = moving_average(price, 10)
#     volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))
#
#     volume = scaler.fit_transform(volume.reshape(-1, 1))
#     price = scaler.fit_transform(price.reshape(-1, 1))
#
#     data = np.concatenate((price, volume), axis=1)
#
#     prediction = SEI_model.predict(np.array([data]))
#
#     prediction = scaler.inverse_transform(prediction)*10000
#
#     result = np.append(scaler.inverse_transform(price)*10000, prediction)
#
#     return result.flatten()
#
#
# # =======================<SOL>=============================
# def sol_prediction():
#     x_step = 50  # length of train_x of each time step
#     dividing = 10000
#
#     dataframe = pyupbit.get_ohlcv(ticker="KRW-SOL", interval="minute1", count=x_step+9)
#
#     volume = dataframe['volume'].values
#     price = dataframe['value'] / dataframe['volume']
#     price = price / dividing
#
#     price = moving_average(price, 10)
#     volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))
#
#     volume = scaler.fit_transform(volume.reshape(-1, 1))
#     price = scaler.fit_transform(price.reshape(-1, 1))
#
#     data = np.concatenate((price, volume), axis=1)
#
#     prediction = SOL_model.predict(np.array([data]))
#
#     prediction = scaler.inverse_transform(prediction)*10000
#
#     result = np.append(scaler.inverse_transform(price)*10000, prediction)
#
#     return result.flatten()
#
#
# # =======================<STX>=============================
# def stx_prediction():
#     x_step = 50  # length of train_x of each time step
#     dividing = 10000
#
#     dataframe = pyupbit.get_ohlcv(ticker="KRW-STX", interval="minute1", count=x_step+9)
#
#     volume = dataframe['volume'].values
#     price = dataframe['value'] / dataframe['volume']
#     price = price / dividing
#
#     price = moving_average(price, 10)
#     volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))
#
#     volume = scaler.fit_transform(volume.reshape(-1, 1))
#     price = scaler.fit_transform(price.reshape(-1, 1))
#
#     data = np.concatenate((price, volume), axis=1)
#
#     prediction = STX_model.predict(np.array([data]))
#
#     prediction = scaler.inverse_transform(prediction)*10000
#
#     result = np.append(scaler.inverse_transform(price)*10000, prediction)
#
#     return result.flatten()
#
#
# # =======================<T>=============================
# def t_prediction():
#     x_step = 50  # length of train_x of each time step
#     dividing = 10000
#
#     dataframe = pyupbit.get_ohlcv(ticker="KRW-T", interval="minute1", count=x_step+9)
#
#     volume = dataframe['volume'].values
#     price = dataframe['value'] / dataframe['volume']
#     price = price / dividing
#
#     price = moving_average(price, 10)
#     volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))
#
#     volume = scaler.fit_transform(volume.reshape(-1, 1))
#     price = scaler.fit_transform(price.reshape(-1, 1))
#
#     data = np.concatenate((price, volume), axis=1)
#
#     prediction = T_model.predict(np.array([data]))
#
#     prediction = scaler.inverse_transform(prediction)*10000
#
#     result = np.append(scaler.inverse_transform(price)*10000, prediction)
#
#     return result.flatten()