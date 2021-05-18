from sklearn.ensemble import RandomForestRegressor
from TS_AutoML.functions import ParameterSearch

# TODO add type hinting


class RandomForest:
    # TODO: Add all RF parameters with default values
    def __init__(self, x_train=None, y_train=None, **config):
        self.x = x_train
        self.y = y_train
        self.n_estimators = config.get('n_estimators', 1000)
        self.n_jobs = config.get('n_jobs', -1)
        self.random_state = config.get('random_state', 0)

    def train(self):
        regr = RandomForestRegressor(n_estimators=self.n_estimators,
                                     n_jobs=self.n_jobs,
                                     random_state=self.random_state)
        # regr = RandomForestRegressor(bootstrap=True,
        #                              max_depth=100,
        #                              max_features=25,
        #                              min_samples_leaf=5,
        #                              min_samples_split=12,
        #                              n_estimators=100)
        return regr.fit(self.x, self.y)

    @staticmethod
    def search_params(df, target_column, grid):
        gridsearch = ParameterSearch(df=df,
                                     target_column=target_column,
                                     regressor=RandomForestRegressor,
                                     grid=grid)
        return gridsearch.optimize()
