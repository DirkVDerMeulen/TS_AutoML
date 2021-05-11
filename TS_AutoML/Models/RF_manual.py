import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error


class TimeSeriesPredictor(RandomForestRegressor):

    def __init__(self, df, predictor=RandomForestRegressor, **config):
        super(TimeSeriesPredictor, self).__init__()
        self.df = df
        self.predictor = predictor
        self.test_window = config.get('test_window', 52)
        self.retrain_frequency = config.get('retrain_frequency', 1)
        self.partitioning_cols = config.get('partitioning_columns')
        self.time_col = config.get('time_column', 'Date')
        self.target_value = config.get('target_value', 'demand')
        self.prediction_results = []
        self.mean_error = []

    def expanding_window_prediction(self):
        for iter, date in enumerate(self.get_prediction_dates()):

            if iter % self.retrain_frequency == 0:  # Only update with update frequency
                train = self.df[self.df[self.time_col] < date]
            test = self.df[self.df[self.time_col] == date]

            X_train, X_test = train.drop(self.target_value, axis=1), test.drop(self.target_value, axis=1)
            y_train, y_test = train[self.target_value].values, test[self.target_value].values

            regressor = self.train_predictor(X_train, y_train)

            prediction = regressor.predict(X_test)

            # Add predictions to df
            X_test['prediction'], X_test['actual'] = list(prediction), list(y_test)
            self.prediction_results.append(X_test)

            error = self.rmsle(y_test, prediction)
            print('Week %d - Error %.5f' % (date, error))
            self.mean_error.append(error)
        print('Mean Error = %.5f' % np.mean(self.mean_error))

        retults = pd.concat(self.prediction_results)
        return retults

    def get_prediction_dates(self):
        return self.df[self.time_col].sort_values(ascending=True).unique()[-self.test_window:]

    def train_predictor(self, X, y):
        regr = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
        return regr.fit(X, y)

    def rmsle(self, ytrue, ypred):
        return np.sqrt(mean_squared_log_error(ytrue, ypred))
