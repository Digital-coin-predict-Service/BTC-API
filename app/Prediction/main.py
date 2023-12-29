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

def prediction():
    matplotlib.use('Agg')

    x_step = 10  # length of train_x of each time step
    y_step = 3  # length of train_y of each time step
    dividing = 10000

    h5_model = "app/Prediction/model.h5"

    model = tf.keras.models.load_model(h5_model)

    dataframe = pyupbit.get_ohlcv(ticker="KRW-BTC", interval="day", count=10)

    data = dataframe['close'].values

    scaling_factor = np.reshape(np.log(np.array([81403000.0, 3619000.0])/dividing), (2, 1))
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
