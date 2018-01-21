#import keras
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from keras.layers import Dense, Activation, LSTM
from keras.models import Sequential

# Create random time series between 2000 and 2018
random.seed(111)
date = pd.date_range(start='2000', periods=209, freq="M")
ts = pd.Series(np.random.uniform(-10, 10, size=len(date)), date).cumsum()
ts.plot()
#plt.show()

TS = np.array(ts)

num_periods = 20
f_horizon = 1

print(TS.shape)
x_data = TS[:(len(TS) - (len(TS) % num_periods))]
x_batches = x_data.reshape(-1, 20, 1)

y_data = TS[1:(len(TS) - (len(TS) % num_periods)) + f_horizon]
y_batches = y_data.reshape(-1, 20, 1)

print(len(x_batches))
print(x_batches.shape)
print(x_batches[0:2])

print(y_batches.shape)
print(y_batches[0:1])


def test_data(series, forecast, num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, 20, 1)
    testY = TS[-(num_periods):].reshape(-1 , 20, 1)
    return testX, testY


X_test, Y_test = test_data(TS, f_horizon, num_periods)
print(X_test.shape)
print(X_test)


model = Sequential()
model.add(LSTM(4, input_shape=(20, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_batches, y_batches, epochs=1, batch_size=20)
