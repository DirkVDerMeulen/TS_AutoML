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
    prediction_end_date=107,
    prediction_lag=1,
    **model_config
)


