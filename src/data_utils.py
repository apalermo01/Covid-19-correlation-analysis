"""
Set of functions to ingest and clean data
"""

import os
import pandas as pd
from tabulate import tabulate
from sodapy import Socrata
from tqdm import tqdm
import numpy as np
from datetime import timedelta
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

def get_policy_data(path = "./data/covid_policies.csv",
                    force_reload = False):
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
    
    if os.path.exists(path) and not force_reload:
        df = pd.read_csv(path, index_col=0)
    else:
        client = Socrata("healthdata.gov", None)
        results = client.get("gyqz-9u7n", limit=10000)
        df = pd.DataFrame.from_records(results)
        df.to_csv(path)
    return df



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
    df=None,
    path = "./data/covid_timeseries.csv",
    clean_path = "./data/covid_timeseries_cleaned.csv",
    force_reload = False,
    force_reclean = False,
):
    """
    Pipeline for cleaning covid case data
    :param df: dataframe of raw uncleaned data
    """

    if df is None:
        if os.path.exists(clean_path) and not force_reclean:
            df = pd.read_csv(clean_path, index_col=0)
            return df

        df = get_covid_data(path, force_reload)

    # restrict date
    max_date = '2021-12-31'
    df = df[df['date'] < max_date]

    # rename some columns
    df = df.rename(columns={
        'cumulative_cases_per_100_000': 'cumulative_cases_1e6', 
        'cumulative_deaths_per_100_000': 'cumulative_deaths_1e6',
        'new_cases_per_100_000': 'new_cases_1e6', 
        'new_deaths_per_100_000': 'new_deaths_1e6', 
        'new_cases_7_day_rolling_avg': 'new_cases_7day', 
        'new_deaths_7_day_rolling_avg': 'new_deaths_7day'
    })

    # drop non-county location types
    df = df.drop(df[~df['location_type'].isin(['county'])].index)

    # convert fips codes to integers
    df['fips_code'] = df['fips_code'].astype(np.int64)

    # handle location / county names
    ### TODO: shouldn't county + state be lowercase?
    df = df.rename(columns={'location_name': 'county'})
    df['county'] = df['county'].str.lower()
    df['full_loc_name'] = df['county'] + ', ' + df['state']

    # remove Puerto Rico and DC
    df = df.drop(df[df['state'].isin(['Puerto Rico', 'District of Columbia'])].index)

    # ensure date is stored as a datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')  

    # fill in missing population data
    # https://data.statesmanjournal.com/census/total-population/total-population-change/chugach-census-area-alaska/050-02063/
    df.loc[(df['county'] == 'chugach') & (df['state'] == 'Alaska'), 'total_population'] = 7102

    # https://data.statesmanjournal.com/census/total-population/total-population-change/copper-river-census-area-alaska/050-02066/
    df.loc[(df['county'] == 'copper river') & (df['state'] == 'Alaska'), 'total_population'] = 2617

    # convert population data to ints
    df['total_population'] = df['total_population'].astype(np.int64)

    ### TODO: might need to handle cumulative_cases/deaths_1e6

    # fill in 0s where new_cases and new_deaths are null
    df = df.fillna(value={'new_cases': 0, 'new_deaths': 0})

    # cast to integers
    df['new_cases'] = df['new_cases'].astype(np.int64)
    df['new_deaths'] = df['new_deaths'].astype(np.int64)

    # handle negatives and nulls in new_cases/deaths_1e6
    df['new_cases_1e6' ].clip(  lower=0, inplace=True)
    df['new_cases_1e6' ].fillna(value=0, inplace=True)
    df['new_deaths_1e6'].clip(  lower=0, inplace=True)
    df['new_deaths_1e6'].fillna(value=0, inplace=True)

    # fill in 0s for 7 day averages for the first few days
    df[['new_cases_7day', 'new_deaths_7day']] = df[['new_cases_7day', 'new_deaths_7day']].mask(
    (df['date'] < pd.to_datetime('2020-01-30', format='%Y-%m-%d')), df[['new_cases_7day', 'new_deaths_7day']].fillna(0))
    
    # re-calculate rolling averages where 7 day average is null
    nulls_case7day  = df[df['new_cases_7day' ].isnull()]
    nulls_death7day = df[df['new_deaths_7day'].isnull()]
    days_7   = timedelta(days=7)

    num_elem = len(nulls_case7day)

    for index, data in tqdm(nulls_case7day.iterrows(),
                            desc="re calculating rolling averages for cases",
                            total=num_elem):
            
        df.loc[index, ['new_cases_7day']] = np.sum(([df['new_cases'][
            (df['full_loc_name']==data.full_loc_name) & 
            ((df['date']<=data.date) & (df['date']>data.date-days_7))
        ].values]))/7

    num_elem = len(nulls_death7day)
    for index, data in tqdm(nulls_death7day.iterrows(),
                            desc='re-calculating rolling averages for deaths',
                            total=num_elem):
            
        df.loc[index, ['new_deaths_7day']] = np.sum(([df['new_deaths'][
            (df['full_loc_name']==data.full_loc_name) & 
            ((df['date']<=data.date) & (df['date']>data.date-days_7))
        ].values]))/7


    # normalize
    df["new_cases_7day_1e6" ] = df['new_cases_7day' ] / (df['total_population']/1e5)
    df["new_deaths_7day_1e6"] = df['new_deaths_7day'] / (df['total_population']/1e5)

    # save to file
    df.to_csv(clean_path)

    return df

def clean_policy_data(
    df,
):
    """
    Pipeline for cleaning covid policy data
    :param df: dataframe of raw uncleaned data
    """