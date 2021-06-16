import pandas as pd
from typing import List

from sklearn.preprocessing import MinMaxScaler


def get_unique_id(df: pd.DataFrame,
                  *args,
                  join_symbol: str,
                  col_name: str = 'ID',
                  drop_cols: bool = False) -> pd.DataFrame:
    """ Creates column with unique ID """

    for col in args:
        df[col] = df[col].astype(str)

    plan_item = df[list(args)].agg(join_symbol.join, axis=1)
    df.insert(0, col_name, plan_item)
    del plan_item

    if drop_cols:
        df.drop(list(args), axis=1, inplace=True)

    return df


# TODO: add sorting on date
def get_lagged_features(
        df: pd.DataFrame,
        *args,
        partitioning_cols: List,
        lags: List
) -> pd.DataFrame:
    """ Gets lagged features from normal feature columns """

    for feat in args:
        for lag in lags:
            df[f'{feat}_lagged_{lag}'] = df.groupby(partitioning_cols)[feat].shift(-lag)

    return df


def scale_data(
        data: pd.DataFrame, minimum: int, maximum: int
) -> pd.DataFrame:
    scaler = MinMaxScaler()
    return data.reshape(minimum, maximum)
