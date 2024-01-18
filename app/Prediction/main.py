import os
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import datetime
from calendar import monthrange
import pyupbit
import time

def moving_average(data, window_size):
    # 주어진 윈도우 크기에 따라 가중치를 생성
    weights = np.repeat(1.0, window_size) / window_size

    # 이동 평균 계산
    moving_avg = np.convolve(data, weights, 'valid')

    return moving_avg


scaler = MinMaxScaler()

def btc_prediction():
    x_step = 50  # length of train_x of each time step
    dividing = 10000

    h5_model = "app/Prediction/BTC_model/model.h5"

    model = tf.keras.models.load_model(h5_model)

    dataframe = pyupbit.get_ohlcv(ticker="KRW-BTC", interval="minute1", count=x_step+9)

    volume = dataframe['volume'].values
    price = dataframe['value'] / dataframe['volume']
    price = price / dividing

    price = moving_average(price, 10)
    volume = np.delete(volume, (1,2,3,4,5,6,7,8,9))

    volume = scaler.fit_transform(volume.reshape(-1, 1))
    price = scaler.fit_transform(price.reshape(-1, 1))

    data = np.concatenate((price, volume), axis=1)

    prediction = model.predict(np.array([data]))

    prediction = scaler.inverse_transform(prediction)*10000

    result = np.append(scaler.inverse_transform(price)*10000, prediction)

    return result.flatten()




def eth_prediction():
    matplotlib.use('Agg')

    x_step = 10  # length of train_x of each time step
    y_step = 3  # length of train_y of each time step
    dividing = 10000

    h5_model = "app/Prediction/ETH_model/model.h5"

    model = tf.keras.models.load_model(h5_model)

    dataframe = pyupbit.get_ohlcv(ticker="KRW-ETH", interval="day", count=10)

    data = dataframe['close'].values

    scaling_factor = np.reshape(np.log(np.array([5805000.0, 94450.0])/dividing), (2, 1))
    scaler = MinMaxScaler().fit(scaling_factor)

    test = data[:x_step]
    test = np.log(test / dividing)
    test = np.reshape(test, (-1, 1))
    test = scaler.transform(test)
    test = np.reshape(test, (1, x_step))

    predict = model.predict(test)
    predict = np.exp(scaler.inverse_transform(predict)
                     .flatten())

    result = np.insert(predict*10000, 0, data[:x_step]).tolist()

    return result


def xrp_prediction():
    matplotlib.use('Agg')

    x_step = 10  # length of train_x of each time step
    y_step = 3  # length of train_y of each time step
    dividing = 10000

    h5_model = "app/Prediction/XRP_model/model.h5"

    model = tf.keras.models.load_model(h5_model)

    dataframe = pyupbit.get_ohlcv(ticker="KRW-XRP", interval="day", count=10)

    data = dataframe['close'].values

    scaling_factor = np.reshape(np.log(np.array([4380.0, 179.0])/dividing), (2, 1))
    scaler = MinMaxScaler().fit(scaling_factor)

    test = data[:x_step]
    test = np.log(test / dividing)
    test = np.reshape(test, (-1, 1))
    test = scaler.transform(test)
    test = np.reshape(test, (1, x_step))

    predict = model.predict(test)
    predict = np.exp(scaler.inverse_transform(predict)
                     .flatten())

    result = np.insert(predict*10000, 0, data[:x_step]).tolist()

    return result
