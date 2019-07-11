# Time Series Forecasting
## Weathernet

Weathernet is a simple LSTM time series forecasting network to predict the temperature of the next day, based on the previous 32 days. Currently it supports only max temperature but can be easily extended to other parameters (e.g. pressure, humidity, etc.

## Data

Data is from the German Weather Service which provides historical climate data for 78 weather stations in Germany. The data is freely available on its website (https://www.dwd.de/DE/leistungen/klimadatendeutschland/klarchivtagmonat.html).
This example is using the data from the Hamburg station (1951 until 2018)

## LSTM Network

The LSTM network is a keras functional model and has 4 Layers (1 Input, 2 LSTM and 1 Dense) and currently supports univariate time-series prediction but could easily be extended to multivariate time-series. 


Layer (type)                 Output Shape              Param #

seq_input (InputLayer)       (1, 32, 1)                0 

cu_dnnlstm_1 (CuDNNLSTM)     (1, 32, 100)              41200

cu_dnnlstm_2 (CuDNNLSTM)     (1, 100)                  80800

output_1 (Dense)             (1, 1)                    101

Total params: 122,101
Trainable params: 122,101
Non-trainable params: 0


## Usage

Project is using MLflow and has 2 entry-points. One for a complete training run followed by a prediction and on for prediction using pre-trained model. 

### Training and predicting
```bash
mlflow run -e train_weathernet.py /home/dominik/ai-platform/tasks/time-series/time-series-forecasting/
```
```bash
mlflow run -e main.py /home/dominik/ai-platform/tasks/time-series/time-series-forecasting/
```



