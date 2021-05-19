import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from TS_AutoML.functions import ParameterSearch

from typing import (
    AnyStr
)

# TODO add type hinting


class RandomForest:
    def __init__(self,
                 x_train: pd.DataFrame,
                 y_train: pd.Series,
                 n_estimators: int = 100,
                 max_depth: int = None,
                 criterion: AnyStr = "mse",
                 min_samples_split: int = 2
                 ):
        self.x = x_train
        self.y = y_train
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split

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
