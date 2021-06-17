import pandas as pd
import numpy as np


from keras.models import Sequential
from keras.layers import (
    Dense,
    LSTM
)

from TS_AutoML.functions import (
    RollingForecast
)

from typing import (
    Dict,
    AnyStr,
    Tuple,
    List,
)

class LstmPredictor:
    def __init__(self,
                 **model_config: Dict
                 ):
        self.input_units = model_config.get('input_units', 100)
        self.n_steps = model_config.get('n_steps')
        self.n_groups = model_config.get('n_groups')
        self.n_features = model_config.get('n_features')
        self.activation = model_config.get('activation_function', 'relu')
        self.optimizer = model_config.get('optimizer', 'adam')
        self.loss = model_config.get('loss', 'mse')

    def get_regressor(self):
        model = Sequential()
        model.add(LSTM(units=self.input_units,
                       activation=self.activation,
                       input_shape=(self.n_steps * self.n_groups, self.n_features)))
        model.add(Dense(self.n_groups))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

    @staticmethod
    def reshape_test_data(input_data):
        shape = [1] + [x for x in input_data.shape]
        input_data = input_data.reshape(*shape)
        return input_data

    @staticmethod
    def train_test_split(df: pd.DataFrame, target_column: AnyStr, date_column: AnyStr, prediction_lag: int,
                         train_steps: int):

        # Move target column to last place in DataFrame
        df.insert(len(df.columns)-1, target_column, df.pop(target_column))

        # Determine number of groups in input data
        n_groups = df[date_column].value_counts().mean()

        # Convert DF to array and save column names
        df = np.array(df)

        def _split_sequences(sequences, n_steps, n_groups):
            X, y = list(), list()

            for iteration in range(int(len(sequences)/ n_groups)):
                # Find end index of the pattern
                end_ix = (iteration + n_steps) * n_groups

                # check if end index beyond dataset
                if end_ix > len(sequences):
                    break

                seq_x = sequences[int(iteration * n_groups):int(end_ix), :-1]
                seq_y = sequences[int(end_ix-n_groups):int(end_ix), -1]

                X.append(seq_x)
                y.append(seq_y)

            return np.array(X), np.array(y)

        # Perform train-test split
        X, y = _split_sequences(df, n_steps=train_steps, n_groups=n_groups)
        X_train, y_train = X[:-prediction_lag - 1], y[:-prediction_lag - 1]
        X_test, y_test = X[-1], y[-1]

        return X_train, y_train, X_test, y_test

    @staticmethod
    def parameter_search():
        pass

    @staticmethod
    def rolling_forecast(
            df: pd.DataFrame,
            groupby: List,
            time_column: AnyStr,
            target_column: AnyStr,
            retrain_frequency: int = 1,
            prediction_start_date: int = 0,
            prediction_end_date: int = 0,
            prediction_lag: int = 1,
            train_epochs: int = None,
            train_batch_size: int = 1,
            train_steps: int = 10,
            **model_params) -> Tuple:
        rolling_forward_predictor = RollingForecast(predictor=LstmPredictor, df=df, groupby=groupby,
                                                    time_col=time_column, target_value=target_column,
                                                    retrain_frequency=retrain_frequency,
                                                    prediction_start_date=prediction_start_date,
                                                    prediction_end_date=prediction_end_date,
                                                    train_steps=train_steps, train_epochs=train_epochs,
                                                    train_batch_size=train_batch_size,
                                                    normalize_data=True, prediction_lag=prediction_lag, **model_params)
        prediction_results, accuracy = rolling_forward_predictor.predict()
        return prediction_results, accuracy