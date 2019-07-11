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
now = datetime.datetime.now()

# Working directories
tensorboard_dir = '/home/dominik/ramdisk/tensorboard/'
csv_dir = '/home/dominik/ramdisk/csv/'
model_filepath = '/home/dominik/Weather/' + str(now) + '/Weather_Net.hf5'
base_dir = '/home/dominik/PycharmProjects/Weather/'

# Load Dataset - Daily climate data Hamburg 1951-01-01 - 2018-12-31
# 0 Mess_Datum - 1 Wind_max - 2 Wind_mittel - 3 Niederschlagshoehe - 4 Sonnenstunden - 5 Schneehoehe - 6 Bedeckung_Stunden - 7 Luftdruck - 8 Temp_mittel - 9 Temp_max - 10 Temp_Min - 11 Temp_Boden - 12 Relative_Feuchte
dataset = np.loadtxt(base_dir + 'hamburg_klima_1951_2018_cleaned.csv', dtype='float32', delimiter=';')
# print dataset
print (dataset.shape)

# Chose which variable to predict - in this case temp_max
single_val_ds = dataset[:, 9].T
single_val_ds = np.reshape(single_val_ds, newshape=(24837, 1))
print (single_val_ds.shape)

# Transform features to range (-1, 1) with sklearn MinMaxScaler
scaler2 = MinMaxScaler(feature_range=(-1, 1))
scaler2.fit(single_val_ds)
scaled_sv_ds = scaler2.transform(single_val_ds)
print (scaled_sv_ds)
dataset = scaled_sv_ds


# Make Backup
def makeBackup():
    os.mkdir('/home/dominik/Weather/' + str(now))
    shutil.copy2('/home/dominik/PycharmProjects/Weather/Weather_Pred.py', '/home/dominik/Weather/' + str(now) + '/Weather_Pred.py')
    print(str(datetime.datetime.utcnow()) + ' - Code saved to /home/dominik/Weather/' + str(now) + '/')


# Create Tensorboard Directory
def create_Tensorboard_dir():
    if os.path.exists(tensorboard_dir) == True:
        shutil.rmtree(tensorboard_dir)
    os.mkdir(tensorboard_dir)


# Create CSV Directory
def create_CSV_dir():
    if os.path.exists(csv_dir) == True:
        shutil.rmtree(csv_dir)
    os.mkdir(csv_dir)


# Create Datafolders
makeBackup()
create_Tensorboard_dir()
create_CSV_dir()
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, batch_size=1, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)


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
    # Set up model
    seq_input = Input(shape=(lookback_window, 1), name='seq_input', batch_shape=(1, lookback_window, 1))
    x = CuDNNLSTM(LSTMunits, kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform', return_sequences=True)(seq_input)
    x = CuDNNLSTM(LSTMunits, kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform', return_sequences=False)(x)
    output_1 = Dense(1, activation='linear', name='output_1')(x)

    # Create model, compile and print summary
    weathernet = Model(inputs=seq_input, outputs=output_1)
    return weathernet

# Fit model to data generator and save it to file
def train_model(weathernet):
    weathernet.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss='mse')
    weathernet.summary()
    weathernet.fit_generator(datagen_train(), steps_per_epoch=steps_per_epoch, workers=10, max_queue_size=100, epochs=200, verbose=2, callbacks=[tensorboard_cb], validation_steps=val_steps, validation_data=datagen_val())
    weathernet.save(filepath=model_filepath)


# Predict on pretrained model
def predict():
    # Load trained keras model
    weather_net = keras.models.load_model('/home/dominik/Weather/2019-07-10 01:14:46.536884/Weather_Net.hf5')

    # Print model summary as sanity check
    weather_net.summary()

    # Run prediction with corresponding data generator
    weather_net_prediction = weather_net.predict_generator(datagen_predict(), steps=predict_steps)
    print (weather_net_prediction.shape)
    print (lookback_window + 2 + steps_per_epoch, lookback_window + 2 + steps_per_epoch + predict_steps)
    ground_truth = np.reshape(dataset[lookback_window + 1 + steps_per_epoch + predict_shift: lookback_window + 1 + steps_per_epoch + predict_steps + predict_shift, 0], newshape=(predict_steps, 1))
    print (ground_truth.shape)
    truth_vs_predict = np.hstack((weather_net_prediction, ground_truth))
    print (truth_vs_predict)

    # Print prediction vs ground truth
    (fig, ax) = plt.subplots()
    ax.plot(scaler2.inverse_transform(ground_truth), color='green', label="Truth")
    ax.plot(scaler2.inverse_transform(weather_net_prediction), color='blue', label="Prediction")

    plt.title('Daily Temperature Hamburg')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
#train()

predict()