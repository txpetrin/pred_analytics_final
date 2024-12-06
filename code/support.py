# Generic Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Declare the constants
TIME_STEPS = 15
PREDICT_STEPS = 5
FEATURES = 1 
ALPHA = 0.001
EPOCHS = 100
EPOCHS_TEST = 100
BATCHES = 30

# Build the RNN model
def build_rnn_model(time_steps, predict_steps, features):
    
    inputs = Input(shape=(time_steps, features), name="Input_Sequence")
    
    lstm = LSTM(256, return_sequences=True, name="LSTM_1", dropout = 0.2)(inputs)
    lstm = LSTM(256, return_sequences=True, name="LSTM_2", dropout = 0.2)(lstm)
    lstm = LSTM(128, return_sequences=False, name="LSTM_4", dropout = 0.2)(lstm)
    
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
    print(scaler.inverse_transform(X[index].reshape(-1,1)))

    single_input = X[index] 

    # Reshape input to (1, 12, 1)
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
