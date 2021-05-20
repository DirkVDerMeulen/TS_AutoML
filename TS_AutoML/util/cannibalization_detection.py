import pandas as pd
import itertools

from scipy.stats import ttest_ind

from typing import (
    List,
    AnyStr
)


class CannibalizationDetector:

    def __init__(self,
                 df: pd.DataFrame,
                 groupby: List,
                 promo_column: AnyStr,
                 date_column: AnyStr,
                 target_value: AnyStr,
                 alpha: float = 0.05):
        self.df = df
        self.groupby = groupby
        self.promo_col = promo_column
        self.date = date_column
        self.target_value = target_value
        self.alpha = alpha

    def detect_cannibalization(self) -> pd.DataFrame:
        # get all combinations of promo item and dependent item
        combinations = self._all_combinations()

        # create DataFrame with all combinations
        combinations = pd.DataFrame(combinations, columns=['dependent', 'on_promo'])
        combinations['cannibalization_rate'] = tuple(zip(combinations.dependent, combinations.on_promo))

        # fill combinations DataFrame with cannibalization rates
        combinations.cannibalization_rate = combinations.cannibalization_rate.apply(
            lambda x: self._detect_dependencies(x[0], x[1]))

        # split into different columns
        combinations['p_value'] = combinations.cannibalization_rate.apply(lambda x: x[0])
        combinations.cannibalization_rate = combinations.cannibalization_rate.apply(
            lambda x: x[2] * x[1] if x[1] else 1)

        for col in ['dependent', 'on_promo']:
            for key in combinations.loc[0, col].keys():
                combinations[f'{col}_{key}'] = combinations[col].apply(lambda x: str(x[key]))
        combinations.drop(['dependent', 'on_promo'], axis=1, inplace=True)

        return combinations

    def _all_combinations(self):
        groups_df = self.df.copy().groupby(self.groupby).size().reset_index()
        groups = [
            {col: groups_df.loc[row, col] for col in self.groupby}
            for row in range(len(groups_df))
        ]
        combinations = list(itertools.permutations(groups, len(self.groupby)))
        return combinations

    def _detect_dependencies(self, promo_item, dependent_item):
        """ two-sample t-test on two series. target column of dependent item when promo item NOT ON promo and target
        column of dependent item when promo item ON promo"""

        # Create DF of Promo demand and Dependent demand
        promo_demand = self.df.loc[(self.df[list(promo_item)] == pd.Series(promo_item)).all(axis=1)]
        dependent_demand = self.df.loc[(self.df[list(dependent_item)] == pd.Series(dependent_item)).all(axis=1)]

        # Create lists of dates on which promo item is and is not on promo
        promo_dates = list(promo_demand[promo_demand[self.promo_col] > 0]['Date'])
        no_promo_dates = list(promo_demand[promo_demand[self.promo_col] == 0]['Date'])

        if not promo_dates:
            return 'no promo', 'no promo', 'no promo'
        else:
            onpromo = dependent_demand[(dependent_demand[self.date].isin(promo_dates)) &
                                       (dependent_demand[self.promo_col] == 0)][self.target_value]
            no_promo = dependent_demand[(dependent_demand[self.date].isin(no_promo_dates)) &
                                        (dependent_demand[self.promo_col] == 0)][self.target_value]

        stat, p = ttest_ind(onpromo, no_promo)
        promo_difference = no_promo.mean() / onpromo.mean()

        if p < self.alpha:
            return p, True, promo_difference
        else:
            return p, False, promo_difference
