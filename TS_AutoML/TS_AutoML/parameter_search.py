import pandas as pd
import numpy as np
import itertools

from sklearn.metrics import mean_squared_error

from typing import (
    Dict,
    Callable,
    AnyStr,
    Tuple,
    List
)


class ParameterSearch:
    def __init__(self,
                 predictor: Callable,
                 df: pd.DataFrame,
                 grid: Dict,
                 time_column: AnyStr,
                 target_column: AnyStr,
                 nr_folds: int,
                 warmup_periods: int,
                 prediction_lag: int):
        self.predictor = predictor
        self.df = df
        self.grid = grid
        self.date = time_column
        self.target = target_column
        self.folds = nr_folds
        self.start_periods = warmup_periods
        self.lag = prediction_lag

    def optimize_parameters(self) -> Tuple:
        """ Main method, perform k-fold validation looping over all parameter combination in the Grid """
        all_names = sorted(self.grid)
        combinations = list(itertools.product(*(self.grid[Name] for Name in all_names)))

        print(f'Training  {self.folds} folds for {len(combinations)} models.')

        results = {}
        iteration = 1
        for combination in combinations:
            params = {key: combination[nr] for nr, key in enumerate(all_names)}

            print(f'Training model {iteration} of {len(combinations)} with parameters: {str(params)}')
            iteration += 1

            error = self._calculate_combination_error(params)
            results[str(params)] = error

        best_combination = max(results, key=results.get)

        return best_combination, results

    def _calculate_combination_error(self, parameters: Dict) -> float:
        """ Method performs k-fold validation and error calculation for a single parameter combination """

        # Get trian-test splits and initial training window
        self._convert_date_to_int()
        train_dates, train_test_sections = self._get_folds()

        # Retrain the model looping over the folds to perform prediction task
        results = []
        for train_sec, test_sec in train_test_sections:
            train_dates = list(set(train_dates + train_sec))

            X_train, y_train, X_test, y_test = self._get_train_test_sets(train_dates=train_dates, test_dates=test_sec)

            regressor = self.predictor(X_train, y_train, **parameters).train()
            out = regressor.predict(X_test)

            result = pd.DataFrame()
            result['prediction'], result['actual'] = list(out), list(y_test)
            results.append(result)

        # Create single DF with results and calculate the error of these results
        results = pd.concat(results)
        results['error_weight'] = results.apply(lambda x: self._mape(x.actual, x.prediction))
        accuracy = sum(results.error_weight) / sum(results.actual)
        return accuracy

    def _convert_date_to_int(self):
        """ Method used to convert Timestamp time_column to integer type time column for easy handling of lag """
        all_dates = sorted(list(self.df[self.date].unique()))
        dates_dict = {all_dates[x]: x for x in range(len(all_dates))}
        self.df[self.date] = self.df[self.date].apply(lambda x: dates_dict[x])

    def _get_folds(self) -> Tuple:
        """Method creates folds of (nearly) equal size for the k-fold validation"""
        all_dates = sorted(list(self.df[self.date].unique()))

        # create list with dates for warm-up period and remove from list of all_dates as they won't be used in val.
        initial_train_dates = all_dates[:self.start_periods]
        del all_dates[:self.start_periods]

        test_dates = all_dates[self.lag:]  # Test period starts after warm-up + lag

        # Loop over test dates to create folds of nearly equal size
        train_test_sections = []
        avg = len(test_dates) / float(self.folds)
        last = 0.0
        while last < len(test_dates):
            test_section = test_dates[int(last): int(last + avg)]
            train_section = [x - len(test_section) - self.lag for x in test_section]

            train_test_sections.append((train_section, test_section))
            last += len(test_section)

        return initial_train_dates, train_test_sections

    def _get_train_test_sets(self, train_dates: List, test_dates: List) -> Tuple:
        """This method uses lists with train- and test dates to perform a train-, test split"""
        train_data = self.df[self.df[self.date].isin(train_dates)].copy()
        test_data = self.df[self.df[self.date].isin(test_dates)].copy()

        X_train, X_test = train_data.copy(), test_data.copy()
        X_train.drop(self.target, axis=1, inplace=True), X_test.drop(self.target, axis=1, inplace=True)
        y_train, y_test = train_data[self.target], test_data[self.target]
        return X_train, y_train, X_test, y_test

    def _mape(self, y_true, y_pred):
        return max(0, ((1 - abs(y_pred-y_true)/y_true) * y_true))
