"""
Set of functions to ingest and clean data
"""

import os
import pandas as pd
from tabulate import tabulate

##################################################################################
### DATA INGESTION ###############################################################
##################################################################################
def get_covid_data(path = "./data/covid_timeseries.csv",
                   force_reload = False):
    """Ingest covid data, pulling it from file or downloading it from the source
    
    Parameters
    ----------
    path : str
        Path to raw data. If force_reload is false, will load data from this path if it exists.

    force_reload : bool
        If true, reloads the data from source regardless of whether or not the data
        has already been downloaded

    Returns
    ----------
    Pandas dataframe with raw covid data
    """

    if os.path.exists(path) and not force_reload:
        df = pd.read_csv(path, index_col=0)
    else:
        df = pd.read_csv('https://query.data.world/s/cxcvunxyn7ibkeozhdsuxub27abl7p')
        df.to_csv(path)
    return df

def get_policy_data(path = "./data/covid_policies.csv"):
    """Ingest covid policy data, pulling it from file or downloading it from the source.

    Parameters
    ----------
    path : str
        Path to raw data. If force_reload is false, will load data from this path if it exists.

    force_reload : bool
        If true, reloads the data from source regardless of whether or not the data
        has already been downloaded

    Returns
    ----------
    Pandas dataframe with raw covid policy data
    """ 

##################################################################################
### DATA EXPLORATION ################################################################
##################################################################################
def eval_df(df,
            print_result = True,
            return_result = False):
    """
    Parameters
    ----------
    df

    print_result

    return_result

    Returns
    --------
    """
    table = []
    assert print_result or return_result, "one of print_result or return_result must be true, otherwise there will be no output!"
    for col in df.columns: 
        table.append([col, df[col].isnull().sum(), set([type(i) for i in df[col].values])])
    
    res = tabulate(table, headers=["field", "num_nulls", "datatypes"])
    if print_result:
        print(res)
    if return_result:
        return res

def check_data(series,
               expect_type='int',
               check_all=True, 
               check_ints=False,
               check_types=False,
               check_negs=True,
               name='Series'):
    """Check that the input array is of the expected datatype, and check for negative values
    
    Parameters
    ------------
    series
        input pandas series
    expect_type
        expected datatype for output message
    check_all
        looks like this checks everything
    check_ints

    check_types
    check_negs
        boolean to check for negative values
    name
        name of column for output
    """

    # check taht all values are integers
    if check_ints: 
        all_ints = lambda ser: all([i.is_integer() for i in ser.values])
        print(f"all decimal components zero? (expect true) {all_ints(series)}")
        return 
    
    if check_types: 
        types = lambda ser: set([type(i) for i in ser.values])
        print(f"datatypes (expect {expect_type}): {types(series)}")
        return 
    
    if check_all: 
        all_ints = lambda ser: all([i.is_integer() for i in ser.values])
        print(f"checking {name}:")
        
        types = set([type(i) for i in series.values])
        print(f"datatypes (expect {expect_type}): {types}")

        nulls = series.isnull().sum()
        print(f"number of nulls: {nulls}")

        if check_negs: 
            num_negative = (series.values < 0).sum()
            print(f"number of negative values: {num_negative}")
        print("=================================================\n")


##################################################################################
### DATA CLEANING ################################################################
##################################################################################
def clean_covid_data(
    df,
):
    """
    Pipeline for cleaning covid case data
    :param df: dataframe of raw uncleaned data
    """


def clean_policy_data(
    df,
):
    """
    Pipeline for cleaning covid policy data
    :param df: dataframe of raw uncleaned data
    """