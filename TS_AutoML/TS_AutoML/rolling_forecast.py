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
                 prediction_lag: int = 1,
                 train_epochs: int = None,
                 train_batch_size: int = 1,
                 normalize_data: bool = False,
                 train_steps: int = 0,
                 **model_params):
        self.predictor = predictor
        self.all_data = df
        self.groupby = groupby
        self.time_col = time_col
        self.start_date = prediction_start_date
        self.end_date = prediction_end_date
        self.target_val = target_value
        self.frequency = retrain_frequency
        self.lag = prediction_lag
        self.epochs = train_epochs
        self.batch_size = train_batch_size
        self.normalize_data = normalize_data
        self.train_steps = train_steps
        self.model_params = model_params
        self.results = []
        self.error = []

    def predict(self):

        # Loop over all dates in test set
        for iter, date in enumerate(self.get_prediction_dates()):

            # Get DF with data available for prediction and test data
            iteration_data = self.all_data[self.all_data[self.time_col] <= date].copy()
            test_data = self.all_data[self.all_data[self.time_col] == date].copy()

            # Normalize data based on method used
            if self.normalize_data:

                # Update normalization data according to frequency
                if iter % self.frequency == 0:
                    normalization_df = iteration_data[iteration_data[self.time_col] < date]

                normalization_values = {col:
                                            {'max': max(normalization_df[col]),
                                             'min': min(normalization_df[col])}
                                        for col in normalization_df.columns}

                for col in iteration_data:
                    iteration_data[col] = (iteration_data[col] - normalization_values[col]['min']) / \
                                          (normalization_values[col]['max'] - normalization_values[col]['min'])

            # Train test splits for respective method used (either df or array output)
            train_test_splits = self.predictor.train_test_split(df=iteration_data, target_column=self.target_val,
                                                                date_column=self.time_col, prediction_lag=self.lag,
                                                                train_steps=self.train_steps)

            # Update train and test data according to update frequency
            if iter % self.frequency == 0:
                X_train, y_train = train_test_splits[0], train_test_splits[1]
            X_test, y_test = train_test_splits[2], train_test_splits[3]
            X_test = self.predictor.reshape_test_data(X_test)

            # Train regressor and predict output
            regressor = self.create_regressor()
            if self.epochs:
                regressor.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
            else:
                regressor.fit(X_train, y_train)
            out = regressor.predict(X_test)

            # Add prediction to test_data and denormalize if necessary
            out = out.reshape(len(test_data))
            test_data['prediction'] = out.tolist()
            test_data['prediction'] = test_data.prediction.clip(0, None)
            if self.normalize_data:
                maximum = normalization_values[self.target_val]['max']
                test_data['prediction'] = test_data['prediction'] * (
                        normalization_values[self.target_val]['max'] - normalization_values[self.target_val]['min']
                ) + normalization_values[self.target_val]['min']

            # Calculate prediction error and error weight
            test_data['prediction_error'] = test_data['prediction'] - test_data[self.target_val]
            test_data['error_weight'] = test_data.apply(lambda x: self._mape(x[self.target_val], x.prediction), axis=1)

            # Add to all results
            self.results.append(test_data)

            # Calculate and print iteration accuracy
            accuracy = sum(test_data.error_weight) / sum(test_data[self.target_val])
            print('Week %d - Accuracy %.5f' % (date, accuracy))

        # Create single DF for all results and calculate accuracy
        results = pd.concat(self.results)
        accuracy = sum(results.error_weight) / sum(results[self.target_val])
        return results, accuracy

    def get_prediction_dates(self):
        # Determine array of dates for which rolling forecast must be done
        prediction_dates = self.all_data[self.time_col].sort_values(ascending=True).unique()
        if self.start_date:
            prediction_dates = prediction_dates[prediction_dates >= self.start_date]
        if self.end_date:
            prediction_dates = prediction_dates[prediction_dates <= self.end_date]
        return prediction_dates

    def create_regressor(self):
        return self.predictor(**self.model_params).get_regressor()

    def _mape(self, y_true, y_pred):
        return max(0, ((1 - abs(y_pred - y_true) / y_true) * y_true))
