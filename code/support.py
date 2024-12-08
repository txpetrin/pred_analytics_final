# Generic Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, Reshape, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsforecast.models import AutoARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsforecast.arima import arima_string

import os
from pathlib import Path

os.chdir("..")

data_path = os.path.join(os.getcwd(), "data")
train_path = os.path.join(data_path, "train2.csv")
submission_path = os.path.join(data_path, "submitformat.csv")

ets_path = os.path.join(data_path, "ets_submission.csv")
arima_path = os.path.join(data_path, "arima_submission.csv")
nn_path = os.path.join(data_path, "nn_submission.csv")
rnn_path = os.path.join(data_path, "rnn_submission.csv")



# Declare the constants
TIME_STEPS = 4
PREDICT_STEPS = 4
FEATURES = 1 
ALPHA = 0.0001
EPOCHS = 50
EPOCHS_TEST = 50
BATCHES = 8

# Build the RNN model
def build_nn_model(time_steps, predict_steps, features):
    
    inputs = Input(shape=(time_steps, features), name="Input_Sequence")
    flatten = Flatten(name="Flatten")(inputs)
    
    dense = Dense(128, name="Dense_1", activation = "relu")(flatten)
    dense = Dense(predict_steps * features, activation="relu", name="DenseFinal")(dense)
    
    outputs = Reshape((predict_steps, features), name="Reshape_Output")(dense)
    
    model = Model(inputs, outputs, name="NN_Predictor")
    return model


# Build the RNN model
def build_rnn_model(time_steps, predict_steps, features):
    
    inputs = Input(shape=(time_steps, features), name="Input_Sequence")
    
    lstm = SimpleRNN(64, return_sequences=False, name="RNN_1")(inputs)
    dense = Dense(predict_steps * features, activation="linear", name="Dense")(lstm)
    
    outputs = Reshape((predict_steps, features), name="Reshape_Output")(dense)
    
    model = Model(inputs, outputs, name="RNN_Predictor")
    return model

def window(observations):
    X = []
    y = []

    for i in range(observations.shape[0] - TIME_STEPS - PREDICT_STEPS + 1):
        X.append(observations[i:i + TIME_STEPS])
        y.append(observations[i + TIME_STEPS:i + TIME_STEPS + PREDICT_STEPS])  # Ensure shape (5, 1)

    X = np.array(X)
    y = np.array(y)

    return X, y


def test_window(X, y, rnn_model, scaler, index):

    single_input = X[index] 

    single_input_reshaped = single_input.reshape(1, TIME_STEPS, 1)

    # Predict 5 future steps
    prediction = rnn_model.predict(single_input_reshaped)

    # Output shape should be (5, 1)
    print(scaler.inverse_transform(prediction.reshape(-1,1)) - scaler.inverse_transform(y[index].reshape(-1,1)))


def test(rnn_model, scaler, test_input):
    single_input = test_input 

    # Reshape input to (1, 12, 1)
    single_input_reshaped = single_input.reshape(1, TIME_STEPS, 1)

    # Predict 5 future steps
    predictions = rnn_model.predict(single_input_reshaped)

    return scaler.inverse_transform(predictions.reshape(-1,1))
