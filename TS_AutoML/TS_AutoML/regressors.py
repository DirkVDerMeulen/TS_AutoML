import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from TS_AutoML.functions import (
    ParameterSearch,
    RollingForecast
)

from typing import (
    Dict,
    AnyStr,
    Tuple,
    List
)


class RandomForest:
    def __init__(self,
                 x_train: pd.DataFrame,
                 y_train: pd.Series,
                 **model_config: Dict
                 ):
        self.x = x_train
        self.y = y_train
        self.n_estimators = model_config.get('n_estimators', 100)
        self.max_depth = model_config.get('max_depth', None)
        self.criterion = model_config.get('criterion', 'mse')
        self.min_samples_split = model_config.get('min_samples_split', 2)

    def train(self):
        regr = RandomForestRegressor(n_estimators=self.n_estimators,
                                     max_depth=self.max_depth,
                                     criterion=self.criterion,
                                     min_samples_split=self.min_samples_split)
        return regr.fit(self.x, self.y)

    @staticmethod
    def parameter_search(df: pd.DataFrame,
                         grid: Dict,
                         time_column: AnyStr,
                         target_column: AnyStr,
                         nr_folds: int,
                         warmup_periods: int,
                         prediction_lag: int) -> Tuple:

        search = ParameterSearch(predictor=RandomForest, df=df, grid=grid, time_column=time_column,
                                 target_column=target_column, nr_folds=nr_folds, warmup_periods=warmup_periods,
                                 prediction_lag=prediction_lag)
        best_fit, all_results = search.optimize_parameters()

        return best_fit, all_results

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
            **model_params) -> Tuple:
        rolling_forward_predictor = RollingForecast(predictor=RandomForest, df=df, groupby=groupby,
                                                    time_col=time_column, target_value=target_column,
                                                    retrain_frequency=retrain_frequency,
                                                    prediction_start_date=prediction_start_date,
                                                    prediction_end_date=prediction_end_date,
                                                    prediction_lag=prediction_lag, **model_params)
        prediction_results, accuracy = rolling_forward_predictor.predict()
        return prediction_results, accuracy
