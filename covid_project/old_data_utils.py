"""
Set of functions to ingest and clean data
"""

import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import us
from sodapy import Socrata
from tabulate import tabulate
from tqdm import tqdm
from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)

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

def get_all_policies(policy_dict: Dict, min_samples: int):
    """Generate an array of all 'full' policy names.
    By 'full' policy name, I mean '<policy> - <start | stop> - <county | state>

    Parameters
    ----------
    policy_dict : Dict
       Dictionary of policy mappings (policy name -> new policy name)

    min_samples : int
        minimum number of times the policy has to be found in the dataset to stay in
        the array

    Returns
    ---------
    np.array containing all policy names
    """
    policy_data = clean_policy_data()
    policy_data_prepped = prep_policy_data(
        policy_data=policy_data, policy_dict=policy_dict, min_samples=min_samples
    )
    all_policies = policy_data_prepped["full_policy"].unique()
    return all_policies


def prep_policy_data(policy_data, policy_dict, min_samples=3):
    """Derives a new policy name in the form <policy name> - <start / stop> - <level>
    df2: DataFrame with the policy data
    policy_dict: dictionary to rename / aggregate policy types
    min_samples: throw out policies that were not implemented many times
    """

    proc_policy_data = policy_data.copy()

    # Replace policies with the ones in policy_dict().
    for key in policy_dict.keys():
        proc_policy_data["policy_type"].replace(
            to_replace=key, value=policy_dict[key], inplace=True
        )

    # Define a new field that includes policy_type, start_stop, and policy_level information
    proc_policy_data.loc[:, "full_policy"] = (
        proc_policy_data["policy_type"]
        + " - "
        + proc_policy_data["start_stop"]
        + " - "
        + proc_policy_data["policy_level"]
    )

    # Get number of times each policy was implemented.
    num_samples = proc_policy_data["full_policy"].value_counts()

    # drop the policy if it was implemented fewer than min_policy times.
    proc_policy_data = proc_policy_data.drop(
        proc_policy_data[
            proc_policy_data["full_policy"].isin(
                num_samples[num_samples.values < min_samples].index
            )
        ].index
    )

    # return the DataFrame
    return proc_policy_data



##################################################################################
### DATA PROCESSING FOR REGRESSION ###############################################
##################################################################################


def get_date_range(date, start_move=0, stop_move=7):
    """Get the date range from date+start_move to date+stop_move"""

    return pd.date_range(
        start=date + timedelta(days=start_move), end=date + timedelta(days=stop_move)
    )



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
    """
    Generate a data prepped for regression analysis

    Parameters
    ----------
    case_data

    policy_data_prepped

    policy_name

    bins_list

    save_path

    save_data

    force_rerun

    pbar

    new_df

    Returns
    --------
    pandas dataframe
    """

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

        
# def prepare_data(case_data: pd.DataFrame,
#                  policy_data_prepped: pd.DataFrame,
#                  bins_list: List,
#                  file_id: str,
#                  policies: Union[None, str, List] = None,
#                  save_path: str = "./data/multi_policy_bins/",
#                  save_data: bool = True,
#                  force_rerun: bool = False,
#                  pbar: bool = True,
#                  new_df: Union[None, pd.DataFrame] = None) -> pd.DataFrame:
#
#     ### reload the dataframe from file if applicable
#     filename = file_id.replace(" - ", "_") +\
#                 "-bins=" + ''.join([str(b[0])+"-"+str(b[1])+"_" for b in bins_list])[:-1] + ".csv"
#
#     if not force_rerun and os.path.exists(save_path + filename):
#         new_df = pd.read_csv(save_path + filename, index_col=0, header=[0, 1])
#         new_df[('info', 'date')] = pd.to_datetime(new_df[('info', 'date')], format='%Y-%m-%d')
#         return new_df
