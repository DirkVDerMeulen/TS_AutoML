import pandas as pd


class MovingAverageForecast:

    def __init__(self, input_df, **config):
        self.df = input_df
        self.window = config['window']
        self.lag = config['lag']
        self.groupby = config['partitioning_columns']
        self.time_column = config['time_column']

    def forecast(self):
        # get all unique combinations for SKU and FcstGroup
        combinations = list(set(self.df[self.groupby].itertuples(index=False, name=None)))

        # Loop over all unique combinations
        results = []
        for SKUID, group in combinations:
            tmp_df = self.df[(self.df.SKUID == SKUID) & (self.df.ForecastGroupID == group)].copy()  # DF for combination

            # Sort values and reset index
            tmp_df.sort_values(by=self.time_column, ascending=True)
            tmp_df.reset_index(drop=True, inplace=True)

            # Create series of target value column
            values = tmp_df['HL_sum']
            windows = values.rolling(self.window + self.lag).sum() - values.rolling(self.lag).sum()
            moving_averages = windows / self.window

            # Add to tmp_df
            tmp_df[f'MA_{self.window}_lag_{self.lag}'] = list(moving_averages)

            results.append(tmp_df)
            del values, windows, moving_averages, tmp_df

        result = pd.concat(results)
        return result
