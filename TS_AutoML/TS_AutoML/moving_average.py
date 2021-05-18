import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

from typing import (
    List,
    AnyStr
)

class MovingAverageForecast:

    def __init__(self, input_df,
                 partitioning_columns: List,
                 time_col: AnyStr,
                 target_series_column: AnyStr,
                 window: int,
                 lag: int,
                 prediction_start_date: int = 0,
                 prediction_end_date: int = 0):
        self.df = input_df
        self.groupby = partitioning_columns
        self.time_column = time_col
        self.target = target_series_column
        self.window = window
        self.lag = lag
        self.start_date = prediction_start_date
        self.end_date = prediction_end_date

    def predict(self):
        # get all unique combinations for SKU and FcstGroup
        combinations = list(set(self.df[self.groupby].itertuples(index=False, name=None)))

        # Loop over all unique combinations
        results = []
        for SKUID, group in combinations:
            tmp_df = self.df[(self.df.SKUID == SKUID) & (self.df.ForecastGroupID == group)].copy()  # DF for combination

            # TODO: adjust to match rolling prediction periods determination
            if self.start_date:
                tmp_df = tmp_df[tmp_df[self.time_column] >= self.start_date - self.lag - self.window + 1]
            if self.end_date:
                tmp_df = tmp_df[tmp_df[self.time_column] <= self.end_date]

            # Sort values and reset index
            tmp_df.sort_values(by=self.time_column, ascending=True)
            tmp_df.reset_index(drop=True, inplace=True)

            # Create series of target value column
            values = tmp_df[self.target]
            windows = values.rolling(self.window + self.lag).sum() - values.rolling(self.lag).sum()
            moving_averages = windows / self.window

            # Add to tmp_df
            tmp_df[f'MA_{self.window}_lag_{self.lag}'] = list(moving_averages)
            tmp_df['prediction_error'] = tmp_df[f'MA_{self.window}_lag_{self.lag}'] - tmp_df[self.target]

            tmp_df.dropna(axis=0, inplace=True)
            results.append(tmp_df)
            del values, windows, moving_averages, tmp_df

        result = pd.concat(results)
        error = self.rmse(result[self.target], result[f'MA_{self.window}_lag_{self.lag}'])
        return result, error

    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
