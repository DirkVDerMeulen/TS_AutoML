import pandas as pd

from sklearn.model_selection import GridSearchCV

from typing import (
    Dict,
    Callable,
    AnyStr,
    Tuple,
    Any
)


class ParameterSearch:
    def __init__(self,
                 df: pd.DataFrame,
                 target_column: AnyStr,
                 regressor: Callable,
                 grid: Dict):
        self.df = df
        self.target = target_column
        self.regressor = regressor
        self.grid = grid

    def optimize(self) -> Tuple[Any, Any]:
        X_train, y_train = self.x_y_split()

        regressor = self.regressor()
        grid_search = GridSearchCV(estimator=regressor,
                                   param_grid=self.grid,
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=2)

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_grid = grid_search.best_estimator_

        return best_params, best_grid

    def x_y_split(self):
        X_train = self.df.copy()
        X_train.drop(self.target, axis=1, inplace=True)
        y_train = self.df[self.target].copy()
        return X_train, y_train
