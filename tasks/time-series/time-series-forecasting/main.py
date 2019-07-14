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

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("city_name", help="Name of the city to get climate data", type=str)
args = parser.parse_args()

# Define Variables
lookback_window = 32
steps_per_epoch = 364
val_steps = 364 * 30
LSTMunits = 100
predict_steps = 364 * 10
predict_shift = 0
city = args.city_name

# Working directories
model_filepath = 'Weather_Net_'+city+'.hf5'
image_dir = 'images_prediction/'

# Load Dataset - Daily climate data Hamburg 1951-01-01 - 2018-12-31
# 0 Mess_Datum - 1 Wind_max - 2 Wind_mittel - 3 Niederschlagshoehe - 4 Sonnenstunden - 5 Schneehoehe - 6 Bedeckung_Stunden - 7 Luftdruck - 8 Temp_mittel - 9 Temp_max - 10 Temp_Min - 11 Temp_Boden - 12 Relative_Feuchte
dataset = np.loadtxt(city+'.csv', dtype='float32', delimiter=';')

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

# Plot prediction vs. ground-truth
def plot_series(yhat, ground_truth):
    (fig, ax) = plt.subplots()
    ax.plot(scaler2.inverse_transform(ground_truth), color='green', label="Truth")
    ax.plot(scaler2.inverse_transform(yhat), color='blue', label="Prediction")
    plt.title('Daily Temperature '+city)
    plt.legend(loc='best')
    plt.savefig(image_dir+city+'_Prediction.png')
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
    keras_model_path = 'Weather_Net_'+city+'.hf5'

    # Load pretrained model
    weathernet = keras.models.load_model(keras_model_path)
    weathernet.summary

    # Make prediction
    yhat = weathernet.predict_generator(datagen_predict(), steps=predict_steps)
    ground_truth = np.reshape(dataset[lookback_window + 1 + steps_per_epoch + predict_shift: lookback_window + 1 + steps_per_epoch + predict_steps + predict_shift, 0], newshape=(predict_steps, 1))
    plot_series(yhat, ground_truth)

    with mlflow.start_run():
        # print out current run_uuid
        run_uuid = mlflow.active_run().info.run_uuid
        print("MLflow Run ID: %s" % run_uuid)
        mlflow.log_artifacts(image_dir, "images")
        mlflow.log_param('City_Name', city)
        mlflow.log_param('Prediction_steps', predict_steps)

