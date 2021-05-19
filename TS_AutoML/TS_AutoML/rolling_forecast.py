import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

from typing import (
    Callable,
    List,
    AnyStr
)


class RollingForecast:
    def __init__(self,
                 predictor: Callable,
                 df: pd.DataFrame,
                 groupby: List,
                 time_col: AnyStr,
                 target_value: AnyStr,
                 periods: int,
                 retrain_frequency: int = 1,
                 **model_params):
        self.predictor = predictor
        self.all_data = df
        self.groupby = groupby
        self.time_col = time_col
        self.target_val = target_value
        self.periods = periods
        self.frequency = retrain_frequency
        self.model_params = model_params
        self.results = []
        self.error = []

    def predict(self):

        # Loop over all dates in test set
        for iter, date in enumerate(self.get_prediction_dates()):

            # Only update training data according to retrain frequency
            if iter % self.frequency == 0:
                train = self.all_data[self.all_data[self.time_col] < date]
            test = self.all_data[self.all_data[self.time_col] == date]

            # Split data into train and test data
            X_train, X_test = train.drop(self.target_val, axis=1), test.drop(self.target_val, axis=1)
            y_train, y_test = train[self.target_val].values, test[self.target_val].values

            # Train regressor and predict output
            regressor = self.train_regressor(X_train, y_train)
            out = regressor.predict(X_test)

            # Add predictions and actuals to input Data
            X_test['prediction'], X_test['actual'] = list(out), list(y_test)
            X_test['prediction_error'] = X_test['prediction'] - X_test['actual']
            self.results.append(X_test)

            error = self.rmse(y_test, out)
            print('Week %d - Error %.5f' % (date, error))

        results = pd.concat(self.results)
        error = self.rmse(results['actual'], results['prediction'])
        return results, error

    def get_prediction_dates(self):
        # TODO: adjust to match MA prediction periods determination
        return self.all_data[self.time_col].sort_values(ascending=True).unique()[-self.periods:]

    def train_regressor(self, X_train, y_train):
        return self.predictor(X_train, y_train, **self.model_params).train()

    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
