import pandas as pd

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
                 retrain_frequency: int = 1,
                 prediction_start_date: int = 0,
                 prediction_end_date: int = 0,
                 **model_params):
        self.predictor = predictor
        self.all_data = df
        self.groupby = groupby
        self.time_col = time_col
        self.start_date = prediction_start_date
        self.end_date = prediction_end_date
        self.target_val = target_value
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
            X_test['error_weight'] = X_test.apply(lambda x: self._mape(x.actual, x.prediction))
            self.results.append(X_test)

            accuracy = sum(X_test.error_weight) / sum(X_test.actual)
            print('Week %d - Accuracy %.5f' % (date, accuracy))

        results = pd.concat(self.results)
        accuracy = sum(results.error_weight) / sum(results.actual)
        return results, accuracy

    def get_prediction_dates(self):
        # Determine array of dates for which rolling forecast must be done
        prediction_dates = self.all_data[self.time_col].sort_values(ascending=True).unique()
        if self.start_date:
            prediction_dates = prediction_dates[prediction_dates >= self.start_date]
        if self.end_date:
            prediction_dates = prediction_dates[prediction_dates <= self.end_date]
        return prediction_dates

    def train_regressor(self, X_train, y_train):
        return self.predictor(X_train, y_train, **self.model_params).train()

    def _mape(self, y_true, y_pred):
        return max(0, ((1 - abs(y_pred-y_true)/y_true) * y_true))
