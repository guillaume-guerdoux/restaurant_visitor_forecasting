import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#  To predict generic time series : https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

df = pd.read_csv('data/international-airline-passengers.csv',
                      usecols=[1])
print(df.info())
df.columns = ['passengers']
df = df/max(df['passengers'])
dataset = df.values
dataset = dataset.astype('float32')

'''plt.plot(dataset)
plt.show()'''
#print(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))


# Create X and Y (Y is just the t+1 value than X)
def create_dataset(dataset, look_back):
    x_data, y_data = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i+look_back), 0]
        x_data.append(a)
        y_data.append(dataset[i + look_back, 0])
    return np.array(x_data), np.array(y_data)

look_back = 3
x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)

x_train = np.reshape(x_train, (x_train.shape[0], 1, 3))
x_test = np.reshape(x_test, (x_test.shape[0], 1, 3))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict
# plot baseline and predictions
'''plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()'''
