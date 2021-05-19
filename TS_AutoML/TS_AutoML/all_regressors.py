import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from TS_AutoML.functions import ParameterSearch

from typing import (
    Dict
)

# TODO add type hinting


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
    def search_params(df, target_column, grid):
        gridsearch = ParameterSearch(df=df,
                                     target_column=target_column,
                                     regressor=RandomForestRegressor,
                                     grid=grid)
        return gridsearch.optimize()
