import numpy as np
import pandas as pd
from datetime import datetime

from TS_AutoML.TS_AutoML.regressors import LstmPredictor

df = pd.read_csv(
    "/Users/dirkvandermeulen/Documents/GitHub/TS_AutoML/TS_AutoML/Testing/assets/demand_week_filtered (1).csv")

df = df[df.category == 'AY']
df = df[['SKUID', 'ForecastGroupID', 'Date', 'HL_sum', 'Promo', 'Carnaval', 'LoadIn']]
df.Date = pd.to_datetime(df.Date, utc=True)
df = df[df.Date > '01-01-2017']
# df = df[(df.SKUID == 109220) & (df.ForecastGroupID == 511704)]
df.sort_values(by=['Date', 'SKUID', 'ForecastGroupID'], inplace=True)
df.insert(len(df.columns)-1, 'HL_sum', df.pop('HL_sum'))

all_dates = sorted(list(set(df.Date)))
date_dict = {all_dates[x]: x for x in range(len(all_dates))}
df.Date = df.Date.apply(lambda x: date_dict[x])

model_config = {
    'n_steps': 52,
    'n_groups': 6,
    'n_features': 6
}

out1, out2 = LstmPredictor.rolling_forecast(
    df=df,
    groupby=['SKUID', 'ForecastGroupID'],
    time_column='Date',
    target_column='HL_sum',
    retrain_frequency=1,
    prediction_start_date=105,
    prediction_lag=107,
    **model_config
)


# for col in ['SKUID', 'ForecastGroupID', 'Promo', 'Date', 'Carnaval', 'LoadIn', 'HL_sum']:
#     mx = df[col].max()
#     df[col] = df[col]/mx
#
# tts = LstmPredictor.train_test_split(df=df,
#                                      target_column='HL_sum',
#                                      date_column='Date',
#                                      prediction_lag=3,
#                                      train_steps=52)
#
# x_train = tts[0]
# y_train = tts[1]
# x_test = tts[2]
# y_test = tts[3]
#
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
#
# model_config = {
#     'n_steps': 52,
#     'n_groups': 6,
#     'n_features': 6
# }
#
# x_test = LstmPredictor.reshape_test_data(x_test)
#
# regressor = LstmPredictor(x_train, y_train, **model_config).train()
#
# regressor.fit(x_train, y_train, epochs=10, verbose=0)
#
# regressor.predict(x_test)
#
# model = Sequential()
# model.add(LSTM(100, activation='relu', input_shape=(52 * 6, 6)))
# model.add(Dense(6, activation='relu'))
# model.compile(optimizer='adam', loss='mse')
# model.fit(x_train, y_train, epochs=20, verbose=0)
#
# pred.predict(pred, input_data=x_test)
