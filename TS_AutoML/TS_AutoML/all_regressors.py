from sklearn.ensemble import RandomForestRegressor


class RandomForest:
    # TODO: Add all RF parameters with default values
    def __init__(self, x_train, y_train, **config):
        self.x = x_train
        self.y = y_train
        self.n_estimators = config.get('n_estimators', 1000)
        self.n_jobs = config.get('n_jobs', -1)
        self.random_state = config.get('random_state', 0)

    def train(self):
        regr = RandomForestRegressor(n_estimators=self.n_estimators,
                                     n_jobs=self.n_jobs,
                                     random_state=self.random_state)
        return regr.fit(self.x, self.y)
