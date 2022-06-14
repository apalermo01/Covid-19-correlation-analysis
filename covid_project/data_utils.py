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
from datetime import datetime
import us
import re
from tqdm import tqdm
#from tqdm.notebook import tqdm

##################################################################################
### DATA INGESTION / RETRIEVAL ###################################################
##################################################################################
def load_covid_data(path = "./data/covid_timeseries.csv",
                   force_reload = False):
    """Loads raw covid data, pulling it from file or downloading it from the source
    
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

def load_policy_data(path = "./data/covid_policies.csv",
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

def get_processed_data(policy_name,
                    bins_list,
                    root_path="./data/single_policy_bins/",):
    """           
    Parameters
    ----------
    policy_name

    bins_list

    root_path

    """

    ### TODO: add option to generate processed data if the file is not present
    filename = policy_name.replace(" - ", "_") +\
                "-bins=" + ''.join([str(b[0])+"-"+str(b[1])+"_" for b in bins_list])[:-1] + ".csv"
    path = root_path + filename

    if os.path.exists(path):
        data = pd.read_csv(root_path + filename, header=[0, 1], index_col=0)
        return True, data
    else:
        return False, None

def get_all_policies(policy_dict,
    min_samples):
    policy_data = clean_policy_data()
    policy_data_prepped = prep_policy_data(policy_data=policy_data,
                                           policy_dict=policy_dict,
                                           min_samples=min_samples)
    all_policies = policy_data_prepped['full_policy'].unique()
    return all_policies


##################################################################################
### DATA EXPLORATION #############################################################
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
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')  
            return df

        df = load_covid_data(path, force_reload)

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
    df = None,
    cleaned_timeseries_path = "./data/covid_timeseries_cleaned.csv",
    path = "./data/covid_policies.csv",
    clean_path = "./data/covid_policies_cleaned.csv",
    force_reload = False,
    force_reclean = False,
):
    """
    Pipeline for cleaning covid policy data
    :param df: dataframe of raw uncleaned data
    """
    
    # get covid policy data
    if df is None:
        if os.path.exists(clean_path) and not force_reclean:
            df = pd.read_csv(clean_path, index_col=0)
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            return df
        
        df = load_policy_data(path, force_reload)
    
    # remove irrelevant columns
    df = df.drop(['geocoded_state', 'comments', 'source', 'total_phases'], axis=1)
    # get covid timeseries data
    timeseries_df = clean_covid_data(
        df=None,
        clean_path = cleaned_timeseries_path
    )

    # clean up state names
    abbr = [elem.abbr for elem in us.states.STATES]
    df = df.drop(df[~df['state_id'].isin(abbr)].index)
    df.replace(to_replace=us.states.mapping('abbr', 'name'), inplace=True)
    df.rename(columns={'state_id': 'state'}, inplace=True)

    ### county
    # convert nulls in count to 'statewide'
    df.fillna(value={'county': 'statewide'}, inplace=True)
    
    # convert to lowercase
    df['county'] = df['county'].str.lower()

    # address mismatches
    county_match = re.compile(" county$")
    munici_match = re.compile(" municipality$")
    city_match = re.compile(" city$")
    Borough_match = re.compile(" borough$")

    df['county'].replace(to_replace=county_match, value='', inplace=True)
    df['county'].replace(to_replace=munici_match, value='', inplace=True)
    df['county'].replace(to_replace=city_match, value='', inplace=True)
    df['county'].replace(to_replace=Borough_match, value='', inplace=True)

    locs = timeseries_df['county'].unique()
    mismatches = [county for county in df['county'][df['county']!='statewide'].unique() 
              if county not in locs]
    assert len(mismatches) == 0, f"[ERROR] found mismatches between timeseries and policy dataset: {mismatches}"

    # fips code
    for index, data in df.iterrows(): 
        if data.policy_level == 'state':
            df.loc[index, 'fips_code'] = np.int64(us.states.lookup(data.state).fips)
    df['fips_code'] = df['fips_code'].astype(np.int64)

    # fix typos in date
    bad_mask = df['date'].str.contains('0020')
    df.loc[bad_mask, 'date'] = ['2020' + elem[4:] for elem in df.loc[bad_mask, 'date'].values]
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.drop(df[(df['date']<min(timeseries_df['date'])) | (df['date']>max(timeseries_df['date']))].index)

    # handle policy names
    policy_replacements_dict = {
        'Stop Initiation Of Evictions Overall Or Due To Covid Related Issues': 'Stop Initiation Of Evictions',
        'Modify Medicaid Requirements With 1135 Waivers Date Of CMS Approval': 'Modify Medicaid Requirements', 
        'Stop Enforcement Of Evictions Overall Or Due To Covid Related Issues': 'Stop Enforcement Of Evictions', 
        'Mandate Face Mask Use By All Individuals In Public Facing Businesses':  'Mandate Face Masks In Businesses', 
        'Mandate Face Mask Use By All Individuals In Public Spaces': 'Mandate Face Masks In Public Spaces', 
        'Reopened ACA Enrollment Using a Special Enrollment Period': 'ACA Special Enrollment Period', 
        'Suspended Elective Medical Dental Procedures': 'Suspend Elective Dental Procedures', 
        'Allow Expand Medicaid Telehealth Coverage': 'Expand Medicaid Telehealth Coverage', 
        'Renter Grace Period Or Use Of Security Deposit To Pay Rent': 'Grace Period / Security Deposit for Rent'
    }

    for key in policy_replacements_dict.keys():
        df['policy_type'].replace(to_replace=key, value=policy_replacements_dict[key], inplace=True)

    df['policy_type'] = df['policy_type'].str.lower()
    policies_drop = ["phase 1", "phase 2", "phase 3", "phase 4", "phase 5", "new phase"]
    df = df.drop(df[df['policy_type'].isin(policies_drop)].index)

    # save
    df.to_csv(clean_path)

    return df




##################################################################################
### DATA PROCESSING ##############################################################
##################################################################################
def get_cases(case_dataframe=None,
              level="county",
              county="orange",
              state="California"):
    
    """ Return the new_case and new_death numbers at the given level of aggregation (county, state, or national). 
    
    Parameters
    ----------
    level : {'county', 'state', 'national'}
        If county, returns a DataFrame filtered to a specific county (default). 
        If state, aggregates the DataFrame to the state level. 
        If national, or any other input, returns the DataFrame aggregated to the national level. 
    county : string 
        desired county
    state : string 
        desired state
    case_dataframe : pandas DataFrame 
        DataFrame to use, case_dataframe by default
    
    Returns 
    ----------
    DataFrame
        case_data filtered to a specific county or aggregated to the state / national level with index=date
    """

    if case_dataframe is None:
        raise NotImplementedError

    # return county level data
    if level == "county":
        return_case_dataframe = case_dataframe[(case_dataframe["county"] == county) & (case_dataframe["state"] == state)].set_index("date")[['new_cases_1e6', 
                                                                                             'new_deaths_1e6', 
                                                                                             'new_cases_7day_1e6', 
                                                                                             'new_deaths_7day_1e6',
                                                                                            ]]

        return return_case_dataframe
    
    # If this is filtered at the state level, filter df to desired state. Otherwise, return national-level data.
    if level == "state": 
        case_dataframe = case_dataframe[case_dataframe['state'] == state]

    # Reindex on location name.
    case_dataframe = case_dataframe.set_index(["full_loc_name"])
    
    # Get a list of all dates.
    all_dates = case_dataframe['date'].unique()

    # Get the total population from the county populations.
    total_population =  sum([(pops / 1e5) for pops in case_dataframe[case_dataframe['date'] == all_dates[0]]['total_population']]) 
    
    # Add up the case and death #s that have the same date.
    new_cases       = [sum([(county_cases / total_population) 
                              for county_cases in case_dataframe[case_dataframe['date'] == dates]['new_cases_1e6']]) 
                              for dates in all_dates]

    new_deaths      = [sum([(county_cases / total_population) 
                              for county_cases in case_dataframe[case_dataframe['date'] == dates]['new_deaths_1e6']]) 
                              for dates in all_dates]

    new_cases_7day  = [sum([(county_cases / total_population) 
                              for county_cases in case_dataframe[case_dataframe['date'] == dates]['new_cases_7day_1e6']]) 
                              for dates in all_dates]

    new_deaths_7day = [sum([(county_cases / total_population) 
                              for county_cases in case_dataframe[case_dataframe['date'] == dates]['new_deaths_7day_1e6']]) 
                              for dates in all_dates]


    return_case_dataframe = pd.DataFrame(data={'date'               : all_dates,
                                   'new_cases_1e6'      : new_cases, 
                                   'new_deaths_1e6'     : new_deaths,
                                   'new_cases_7day_1e6' : new_cases_7day,
                                   'new_deaths_7day_1e6': new_deaths_7day
                                   }).set_index(["date"]) 
    return return_case_dataframe

def get_policies(policy_dataframe=None,
                    state="California",
                    county="statewide",
                    state_policies=True,
                    county_policies=True):
       
    """Get the policy data at county level, state level, or both.  

    Parameters
    ----------
    state : string 
        selected state
    county : string
        selected county
    state_policies : boolean 
        include policies at the state level (default: True)
    county_policies : boolean 
        include policies at the county level (default: True)
    
    Returns
    ---------- 
    filtered DataFrame
    """
    
    if policy_dataframe is None:
        raise NotImplementedError

    # state AND county policies
    if state_policies and county_policies:

        return policy_dataframe[(policy_dataframe['state'] == state) &
                  ((policy_dataframe["county"] == county) | (policy_dataframe["county"] == "statewide"))]
    
    # state policies only
    elif state_policies and not county_policies: 
        return policy_dataframe[ (policy_dataframe['state'] == state) & (policy_dataframe["county"] == "statewide")]
    
    # county policies only
    else:  
         return policy_dataframe[ (policy_dataframe['state'] == state) & (policy_dataframe["county"] == county)]

def generate_state_case_dict(case_df,
                             save_data=False,
                             load_from_file=False,
                             path="./data/state_data/"):
    """Generate a dictionary of dataframes with statewide case data for every state

    Parameters
    ----------
    case_df

    save_data

    load_from_file

    path

    Returns
    --------
    dictionary
    """

    state_cases_dict = dict()
    for state in tqdm([elem.name for elem in us.states.STATES]): 
        state_save_path = path+str(state)+".csv"

        if load_from_file and os.path.exists(state_save_path):
            state_data = pd.read_csv(state_save_path, index_col=0)
        else:
            state_data = get_cases(case_dataframe=case_df,
                                            level="state",
                                            state=state);
            if save_data:
                if not os.path.exists(path):
                    os.makedirs(path)
                state_data.to_csv(state_save_path)
        state_cases_dict[state] = state_data
    return state_cases_dict

def calculate_deltas(case_df,
                     policy_df,
                     measure_period=14,
                     filtered_policies=None,
                     state_cases_dict=None,
                     save_state_data=True,
                     load_state_data_from_file=True,
                     state_data_path="./data/state_data/",
                     results_path="./data/deltas/",
                     force_run=False,
                     save_results=True): 
    """For every policy implementation at the state and county level, calculate the change in case and death numbers. 
    
    Parameters
    ---------- 
    measure_period : int
        time to wait (in days) before measuring a change in new case or death numbers (14 by default)
    filtered_policies : array-like 
        specify policies to select (defaul: None- calulate deltas for all policies)
    case_df : pandas DataFrame
        DataFrame with case / death information (default: df)
    policy_df : pandas DataFrame
        DataFrame with police information (default: df2)
    
    Returns
    ----------  
    A copy of the covid policies df (df2) with 2 appended columns for the change in case and death numbers. 
    """
    
    # Load all state-aggregated datasets into a dictionary. We expect to need all 50 states so let's take the time to aggregate
    # the state data now so we don't need to do it repeatedly in the loop. 
    
    if state_cases_dict is None: 
        state_cases_dict = generate_state_case_dict(case_df=case_df,
                                                    save_data = save_state_data,
                                                    load_from_file = load_state_data_from_file,
                                                    path = state_data_path)

    results_file = f"{results_path}{str(measure_period)}_delta.csv"
    if os.path.exists(results_file) and not force_run:
        results = pd.read_csv(results_file, index_col=0)
        return results, state_cases_dict

    # Initialize wait period before measurement.
    wait_period = timedelta(days=measure_period)
    day_1 = timedelta(days=1)
    
    def sub_calc_deltas(ser, date, wait=wait_period): 
        """Wrap repeated calculations in a sub function to avoid repetition."""
        day_1 = timedelta(days=1)
        ser.index = pd.to_datetime(ser.index, format="%Y-%m-%d")
    
        start      = ser[ser.index==date].values[0]
        start_1day = ser[ser.index==date+day_1].values[0]
        
        end        = ser[ser.index==date+wait].values[0]
        end_1day   = ser[ser.index==date+wait+day_1].values[0]
        
        return [start, start_1day, end, end_1day]
    
    # If there we are only examining select policies, then filter those out.
    if filtered_policies is not None: 
        policy_df = policy_df.loc[policy_df['policy_type'].isin(filtered_policies)]
        
    correlated_df = policy_df.copy()
    
    # Initially fill the delta column with nan.
    correlated_df.loc[:, f"case_{measure_period}_day_delta"] = np.nan
    correlated_df.loc[:, f"case_{measure_period}_day_accel"] = np.nan
    correlated_df.loc[:, f"death_{measure_period}_day_delta"] = np.nan
    correlated_df.loc[:, f"death_{measure_period}_day_accel"] = np.nan
    


    #case_df['date'] = datetime.strptime(case_df['date'].values, "%Y-%m-%d")
    case_df['date'] = pd.to_datetime(case_df['date'], format="%Y-%m-%d")
    case_df = case_df.set_index('date')
    total_policies = len(policy_df)
    
    for index, data in tqdm(policy_df.iterrows()): 
        data['date'] = datetime.strptime(data['date'], "%Y-%m-%d")

        # If this is a state-level policy, then we already have the DataFrame to use. 
        if data.policy_level == 'state': 
            state_df = state_cases_dict[data.state]
            ser_cases = state_df['new_cases_7day_1e6' ]
            ser_deaths = state_df['new_deaths_7day_1e6']
        
        # This must be a county level policy- filter the appropriate data. 
        else:
            ser_cases = case_df['new_cases_7day_1e6' ][case_df['fips_code'] == data.fips_code]
            ser_deaths = case_df['new_deaths_7day_1e6'][case_df['fips_code'] == data.fips_code]
        

        # Get the case and death numbers at the appropriate days. 
        c11, c12, c21, c22 = sub_calc_deltas(ser_cases, date=data.date)
        d11, d12, d21, d22 = sub_calc_deltas(ser_deaths, date=data.date)
           
        # Calculate the difference in new cases at the selected dates. 
        correlated_df.at[index, f"case_{measure_period}_day_delta"] = c21 - c11
        correlated_df.at[index, f"death_{measure_period}_day_delta"] = d21 - d11
        
        # Calculate the change in curvature (aka acceleration) of the case / death plots at policy implementation and
        # measure_period days afterwards. 
        
        correlated_df.at[index, f"case_{measure_period}_day_accel"] = ((c12-c11) - (c21-c22)) / measure_period
        correlated_df.at[index, f"death_{measure_period}_day_accel"] = ((d12-d11) - (d21-d22)) / measure_period    
    
    if save_results:
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        correlated_df.to_csv(results_file)
    return correlated_df, state_cases_dict

def calc_delta_stats(deltas, measure_period=14, min_samples=10):
    """Take the deltas calculated with each policy and calculate the average and sd. 
    Parameters
    ---------- 
    deltas : pandas DataFrame 
        dataframe of policy deltas on which to do the calculations
    measure_period : int 
        time to wait (in days) before measuring a change in new case or death numbers (14 by default)
    min_samples : int 
        minimum number of samples that a policy must have for reporting of average and std (default: 10)
    
    Returns
    ----------   
    A dataframe with a record for the start/stop of each policy type and the average / std of the change in 
    case / death numbers measure_period days after implementation
    """
    # Generate a new list of policy types differentiating between start and stop. 
    policy_types = ([elem + " - start" for elem in deltas['policy_type'].unique()]  
                    + [elem + " - stop"  for elem in deltas['policy_type'].unique()])
    
    # Initialize empty arrays for the associated statistics.
    case_avg, death_avg, case_std, death_std, num_samples = [], [], [], [], []
    case_accel_avg, death_accel_avg, case_accel_std, death_accel_std = [], [], [], []
    
    # Loop through all the policy types.
    for policy in policy_types:
        
        # Determine whether this policy is the beginning or end.  
        if policy.endswith("stop"):
            len_index = -7
            start_stop = "stop"
        else: 
            len_index = -8
            start_stop = "start"
        
        # Get arrays of all the deltas for each type of policy 
        case_data  = deltas[(deltas['policy_type'] == policy[:len_index]) & 
                            (deltas['start_stop'] == start_stop)][f'case_{measure_period}_day_delta']

        death_data = deltas[(deltas['policy_type'] == policy[:len_index]) & 
                            (deltas['start_stop'] == start_stop)][f'death_{measure_period}_day_delta']

        case_accel_data = deltas[(deltas['policy_type'] == policy[:len_index]) & 
                                 (deltas['start_stop'] == start_stop)][f'case_{measure_period}_day_accel']
         
        death_accel_data = deltas[(deltas['policy_type'] == policy[:len_index]) & 
                                  (deltas['start_stop'] == start_stop)][f'death_{measure_period}_day_accel']

        num_samples.append(len(case_data))
        
        # Calculate the averages and standard deviations for each policy
        case_avg.append(np.nanmean(case_data.to_numpy()))
        death_avg.append(np.nanmean(death_data.to_numpy()))
            
        case_std.append(np.nanstd(case_data.to_numpy()))
        death_std.append(np.nanstd(death_data.to_numpy()))
        
        case_accel_avg.append(np.nanmean(case_accel_data.to_numpy()))
        death_accel_avg.append(np.nanmean(death_accel_data.to_numpy()))
        
        case_accel_std.append(np.nanstd(case_accel_data.to_numpy()))
        death_accel_std.append(np.nanstd(death_accel_data.to_numpy()))
        
        
        
    # Construct the dataframe to tabulate the data.
    delta_stats = pd.DataFrame(np.transpose([case_avg, case_accel_avg, death_avg, death_accel_avg, 
                                             case_std, case_accel_std, death_std, death_accel_std, 
                                             num_samples]), index=policy_types, 
                               columns=['case_avg', 'case_accel_avg', 'death_avg', 'death_accel_avg', 
                                        'case_std', 'case_accel_std', 'death_std', 'death_accel_std', 
                                        'num_samples']
                              )

    # Drop record with less than min_samples samples.
    delta_stats.drop(delta_stats[delta_stats['num_samples'] <= min_samples].index, inplace=True)
    
    return delta_stats

##################################################################################
### DATA PROCESSING FOR REGRESSION ###############################################
##################################################################################
def prep_policy_data(policy_data,
                     policy_dict,
                     min_samples=3):
    """Small funciton to process policy data
    df2: DataFrame with the policy data
    policy_dict: dictionary to rename / aggregate policy types
    min_samples: throw out policies that were not implemented many times
    """
    
    proc_policy_data = policy_data.copy()
    
    # Replace policies with the ones in policy_dict(). 
    for key in policy_dict.keys():
        proc_policy_data['policy_type'].replace(to_replace=key, value=policy_dict[key], inplace=True)
        
    # Define a new field that includes policy_type, start_stop, and policy_level information
    proc_policy_data.loc[:, 'full_policy'] = proc_policy_data['policy_type'] + " - " +\
                                        proc_policy_data['start_stop'] + " - " +\
                                        proc_policy_data['policy_level']
    
    # Get number of times each policy was implemented.
    num_samples = proc_policy_data['full_policy'].value_counts()
    
    # drop the policy if it was implemented fewer than min_policy times. 
    proc_policy_data = proc_policy_data.drop(proc_policy_data[
        proc_policy_data['full_policy'].isin(num_samples[num_samples.values < min_samples].index)
    ].index)
    
    # return the DataFrame
    return proc_policy_data

def prepare_new_df(case_data):
    """Initialize the new dataframe"""

    dependent_vars = [
        'new_cases_1e6',
        'new_deaths_1e6',
        'new_cases_7day_1e6',
        'new_deaths_7day_1e6',
    ]

    tuples_info = [('info', 'location_type'),
               ("info", "state"),
               ("info", "county"),
               ("info", "date"),]

    dependent_cols = [("info", e) for e in dependent_vars]
    tuples_info = tuples_info + dependent_cols
    case_data_cols = ['location_type', 'state', 'county', 'date']
    case_data_cols = case_data_cols + dependent_vars

    info_cols = pd.MultiIndex.from_tuples(tuples_info)
    new_df = pd.DataFrame(columns = info_cols)
    new_df[tuples_info] = case_data[case_data_cols]
    
    return new_df


def prepare_data(case_data,
                 policy_data_prepped,
                 policy_name,
                 bins_list,
                 save_path = "./data/single_policy_bins/",
                 save_data = True,
                 force_rerun = False,
                 pbar = True,
                 new_df = None):

    def get_date_range(date, start_move=0, stop_move=7): 
        """Get the date range from date+start_move to date+stop_move"""

        return pd.date_range(start=date+timedelta(days=start_move), 
                             end=date+timedelta(days=stop_move))
    
    ### reload the dataframe from file if applicable
    filename = policy_name.replace(" - ", "_") +\
                "-bins=" + ''.join([str(b[0])+"-"+str(b[1])+"_" for b in bins_list])[:-1] + ".csv"
    
    if not force_rerun and os.path.exists(save_path + filename):
        new_df = pd.read_csv(save_path + filename, index_col=0, header=[0, 1])
        new_df[('info', 'date')] = pd.to_datetime(new_df[('info', 'date')], format='%Y-%m-%d')
        return new_df
    
    ### initialize the new dataframe
    if new_df is None:
        new_df = prepare_new_df(case_data)

    tuples_policies = [ (policy_name, (str(date_range[0]) + "-" + str(date_range[1])))
                           for date_range in bins_list]    
    cols_polices = pd.MultiIndex.from_tuples(tuples_policies)
    policies_df = pd.DataFrame(columns=cols_polices)
    new_df = pd.concat([new_df, policies_df])
    new_df = new_df.fillna(0)
    policy_data_filtered = policy_data_prepped[policy_data_prepped['full_policy']==policy_name]

    # generate dataframe
    df_dict = policy_data_filtered.to_dict('records')
    for row in tqdm(df_dict, disable=not pbar):
        for date_bin in bins_list:
            date_range = get_date_range(row['date'], date_bin[0], date_bin[1])

            # Generate label (this is the 2nd level label in the multiIndexed column)
            label = (str(date_bin[0]) + "-" + str(date_bin[1]))
            new_df.loc[(new_df[('info', 'date')].isin(date_range)) &\
                       ((new_df[('info', 'county')] == row['county']) | (row['policy_level'] == 'state')) &\
                       (new_df[('info', 'state')] == row['state']), (policy_name, label)] = 1

    if save_data:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        new_df.to_csv(save_path+filename)
    return new_df

def generate_dataset_group(bins_list,
                           policy_dict,
                           min_samples=3):
    """Generate datasets for every policy for a given group of bins
    Parameters
    ----------
    bins_list
    
    policy_dict
    """

    case_data = clean_covid_data()
    policy_data = clean_policy_data()
    
    policy_data_prepped = prep_policy_data(policy_data=policy_data,
                                           policy_dict=policy_dict,
                                           min_samples=min_samples)
    
    all_policies = policy_data_prepped['full_policy'].unique()
    new_df = prepare_new_df(case_data)

    for policy in tqdm(all_policies, desc='generating datasets for policies'):
        prepare_data(
            case_data=case_data,
            policy_data_prepped = policy_data_prepped,
            policy_name = policy,
            bins_list = bins_list,
            pbar = False,
            new_df=new_df
        )