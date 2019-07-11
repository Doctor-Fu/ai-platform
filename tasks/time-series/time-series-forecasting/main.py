# Imports
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Model
from keras.layers import Dense, CuDNNLSTM, Input
import datetime
import os
import shutil
from numpy.random import seed
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
import argparse
import mlflow.sklearn
from mlflow import log_metric
import mlflow.keras
import warnings
import sys

# Initialize
# Keras and Tensorflow Random Numbers
seed(1)
set_random_seed(2)

# Clear
# Previous Tensorflow Session
tf.keras.backend.clear_session()

# Define Variables
lookback_window = 32
steps_per_epoch = 364
val_steps = 364 * 30
LSTMunits = 100
predict_steps = 364 * 10
predict_shift = 0

# Working directories
model_filepath = 'Weather_Net.hf5'

# Load Dataset - Daily climate data Hamburg 1951-01-01 - 2018-12-31
# 0 Mess_Datum - 1 Wind_max - 2 Wind_mittel - 3 Niederschlagshoehe - 4 Sonnenstunden - 5 Schneehoehe - 6 Bedeckung_Stunden - 7 Luftdruck - 8 Temp_mittel - 9 Temp_max - 10 Temp_Min - 11 Temp_Boden - 12 Relative_Feuchte
dataset = np.loadtxt('hamburg_climate_1951_2018.csv', dtype='float32', delimiter=';')

# Chose which variable to predict - in this case 9 - Temp_max
variable_to_forecast = 9
single_val_ds = dataset[:, variable_to_forecast].T
single_val_ds = np.reshape(single_val_ds, newshape=(24837, 1))
print(single_val_ds.shape)

# Transform features to range (-1, 1) with sklearn MinMaxScaler
scaler2 = MinMaxScaler(feature_range=(-1, 1))
scaler2.fit(single_val_ds)
scaled_sv_ds = scaler2.transform(single_val_ds)
print(scaled_sv_ds)
dataset = scaled_sv_ds

# Fit generators for keras NN - sliding window
# Training Generator
def datagen_train():
    while 1:
        for i in range(lookback_window, lookback_window + steps_per_epoch + 1):
            train_x = dataset[i - lookback_window:i, 0]
            train_x_reshape = np.reshape(train_x, newshape=(1, lookback_window, 1))
            train_y = dataset[i + 1, 0]
            train_y_reshape = np.reshape(train_y, newshape=(1, 1))
            yield ({'seq_input': train_x_reshape}, {'output_1': train_y_reshape})

# Validation Generator
def datagen_val():
    while 1:
        for i in range(lookback_window + steps_per_epoch, lookback_window + steps_per_epoch + val_steps + 1):
            train_x = dataset[i - lookback_window:i, 0]
            train_x_reshape = np.reshape(train_x, newshape=(1, lookback_window, 1))
            train_y = dataset[i + 1, 0]
            train_y_reshape = np.reshape(train_y, newshape=(1, 1))
            yield ({'seq_input': train_x_reshape}, {'output_1': train_y_reshape})

# Prediction Generator
def datagen_predict():
    while 1:
        for i in range(lookback_window + steps_per_epoch, lookback_window + steps_per_epoch + val_steps + 1):
            # print i
            train_x = dataset[i - lookback_window:i, 0]
            train_x_reshape = np.reshape(train_x, newshape=(1, lookback_window, 1))
            train_y = dataset[i + 1, 0]
            train_y_reshape = np.reshape(train_y, newshape=(1, 1))
            yield ({'seq_input': train_x_reshape}, {'output_1': train_y_reshape})

# Train LSTM sequence prediction model
def build_and_compile_model():
    seq_input = Input(shape=(lookback_window, 1), name='seq_input', batch_shape=(1, lookback_window, 1))
    x = CuDNNLSTM(LSTMunits, kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform', return_sequences=True)(seq_input)
    x = CuDNNLSTM(LSTMunits, kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform', return_sequences=False)(x)
    output_1 = Dense(1, activation='linear', name='output_1')(x)

    weathernet = Model(inputs=seq_input, outputs=output_1)

    return weathernet

# Fit model to data generator and save it to file
def train_model(weathernet):
    weathernet.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss='mse')
    weathernet.summary()
    weathernet.fit_generator(datagen_train(), steps_per_epoch=steps_per_epoch, workers=10, max_queue_size=100, epochs=200, verbose=2, callbacks=[tensorboard_cb], validation_steps=val_steps, validation_data=datagen_val())
    weathernet.save(filepath=model_filepath)

# Plot prediction vs. ground-truth
def plot_series(yhat, ground_truth):
    (fig, ax) = plt.subplots()
    ax.plot(scaler2.inverse_transform(ground_truth), color='green', label="Truth")
    ax.plot(scaler2.inverse_transform(yhat), color='blue', label="Prediction")
    plt.title('Daily Temperature Hamburg')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Initialize
    # Keras and Tensorflow Random Numbers
    seed(1)
    set_random_seed(2)

    # Clear
    # Previous Tensorflow Session
    tf.keras.backend.clear_session()

    # Set pretrained model
    keras_model_path = sys.argv[1] if len(sys.argv) > 1 else 'Weather_Net.hf5'

    # Load pretrained model
    weathernet = keras.models.load_model(keras_model_path)
    weathernet.summary

    # Define city data file
    city_name = sys.argv[2] if len(sys.argv) > 2 else 'hamburg_climate_1951_2018.csv'

    # Make prediction
    yhat = weathernet.predict_generator(datagen_predict(), steps=predict_steps)
    ground_truth = np.reshape(dataset[lookback_window + 1 + steps_per_epoch + predict_shift: lookback_window + 1 + steps_per_epoch + predict_steps + predict_shift, 0], newshape=(predict_steps, 1))
    plot_series(yhat, ground_truth)
    with mlflow.start_run():
        #mlflow.log_param("keras_model_path", keras_model_path)
        #mlflow.log_param("city_name", city_name)

        # mlflow.log_artifact(output_photo_name, "output_photo_name")

