import pandas as pd
from typing import List, Tuple, Optional, Union

def prep_sf_format(
    X: pd.DataFrame,
    target_col: str,
    date_col: Optional[str] = None,
    freq: str ='MS',
    filter: Optional[Tuple] = None
    ):
    """_summary_

    Args:
        X (pd.DataFrame): input data (timeseries)
        target_col (str): column with target timeseries (y)
        date_col (Optional[str], optional): Defaults to None. column with date, optional. If None, index is created basing on the freq
        freq (str): Defaults to 'MS'. Frequency to be used for the index
        filter (Optional[Tuple], optional): Defaults to None. if not None:
            filter is applied to the input data
            filter[0] is the column name(s):
            filter[1] is the value(s)
    Returns:
        pd.DataFrame: with columns ['unique_id', 'ds', 'y'] where unique_id target_col name or target_col + filter[1] if filter is not None
    """
    assert target_col in X.columns, f"target_col {target_col} not in X.columns"
    assert date_col is None or date_col in X.columns, f"date_col {date_col} not in X.columns"
    assert freq in ['H', 'D', 'W', 'M', 'MS', 'Q', 'Y'], f"freq {freq} not in ['H', 'D', 'W', 'M', 'MS', 'Q', 'Y']"
    if filter is not None and isinstance(filter[0], list):
        assert len(filter[0]) == len(filter[1]), f"filter[0] and filter[1] should be of the same length"
    elif filter is not None and not isinstance(filter[0], list):
        filter = ([filter[0]], [filter[1]])
    
    X = X.copy()
    #TODO chceck if filter[0] is a list, if true, then filter[1] should be a list of the same length.
    # then geneate query string and apply it to X

    unique_id = f"{target_col}"
    query = []
    for col, val in zip(filter[0], filter[1]):
        # TODO make format 01, 02, 03, for stores and items
        qry = f"({col} == {val})" if not isinstance(val, str) else f"({col} == '{val}')"
        unique_id += f"_{val}"
        query.append(qry)
    
    query = ' & '.join(query)   
    
    if filter is not None:
        X = X.query(query)
    if date_col is not None:
        X = X[[date_col, target_col]]
    else:
        X['ds'] = pd.date_range(start=X.index[0], periods=len(X), freq=freq)
        X = X[['ds', target_col]]
    
    X['unique_id'] = unique_id
    
    X.columns = ['ds', 'y', 'unique_id']
    X['ds'] = pd.to_datetime(X['ds'])
    # TODO change frequency of ds to freq, i.e. aggregate over ds to a certain frequency
    
    return X[['unique_id', 'ds', 'y']].groupby(['unique_id', 'ds']).sum().reset_index()

def synchronize_yX(y: pd.DataFrame, X: pd.DataFrame, left_on: str = 'ds', right_on: str = 'ds'):
    
    assert isinstance(y, pd.DataFrame), f"y should be a pd.DataFrame, got {type(y)}"
    assert isinstance(X, pd.DataFrame), f"X should be a pd.DataFrame, got {type(X)}"
    assert left_on in y.columns, f"y should have column {left_on}"
    assert right_on in X.columns, f"X should have column {right_on}"
    
    yx = pd.merge(y, X, left_on=left_on, right_on=right_on, how='outer')
    yx = yx.sort_values(by=['ds'])
    
    return yx

# TODO: add function to prepare X for prophet
def prepare_X(
    X: pd.DataFrame,
    target_col: Optional[List[str]] = None, # columns to be used as target in none use all columns but date_col
    date_col: Optional[str] = None,
    freq: str ='MS',
    filter: Optional[Tuple] = None # filter is used the same as in prep_sf_format
    ):
    
    pass
