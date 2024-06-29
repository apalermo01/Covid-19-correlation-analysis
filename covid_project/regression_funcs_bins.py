"""
Functions for notebooks 5 and 6 - linear regression
"""

import pandas as pd
from covid_project.data_cleaning import clean_covid_data, clean_policy_data
from typing import List, Union
import statsmodels.api as sm
from tqdm.auto import tqdm
import os
from datetime import timedelta
import json

pd.set_option('future.no_silent_downcasting', True)

def generate_dataset_group_single_policy(
                           bins_list,
                           policy_dict,
                           min_samples=3):
    """Generate datasets for every policy for a given group of bins
    Parameters
    ----------
    bins_list
    
    policy_dict

    min_samples
    """

    case_data = clean_covid_data()
    policy_data = clean_policy_data()
    
    policy_data_prepped = prep_policy_data(policy_data=policy_data,
                                           policy_dict=policy_dict,
                                           min_samples=min_samples)
    
    all_policies = policy_data_prepped['full_policy'].unique()
    new_df = prepare_new_df(case_data)

    for policy in tqdm(all_policies, desc='generating datasets for policies'):
        prepare_data_single_policy(
            case_data=case_data,
            policy_data_prepped = policy_data_prepped,
            policies = policy,
            bins_list = bins_list,
            pbar = False,
            new_df=new_df)


def prep_policy_data(policy_data,
                     policy_dict,
                     min_samples=10,):
    """Preprocess the policy data for Machine Learning models.

    Parameters
    ------------
    df2: DataFrame
            policy data
    policy_dict: dictionary
            Dictionary defined in policy_dict.py to rename and aggregate policies.
    min_samples: integer
            Throw out policies that were not implemented min_samples times.

    Returns
    ----------
    proc_policy_data: DataFrame
            The preprocessed policy data
    """

    # Replace policies with the ones in policy_dict().
    policy_data['policy_type'] = policy_data['policy_type'].replace(policy_dict)

    # Define a new field that includes policy_type, start_stop,
    # and policy_level information
    policy_data.loc[:, 'full_policy'] =\
    policy_data['policy_type'] + " - " +\
    policy_data['start_stop'] + " - " +\
    policy_data['policy_level']

    proc_policy_data = policy_data.copy()

    # Get number of times each policy was implemented.
    num_samples = proc_policy_data['full_policy'].value_counts()

    # drop the policy if it was implemented fewer than min_samples times.
    proc_policy_data = proc_policy_data.drop(proc_policy_data[
        proc_policy_data['full_policy'].isin(
                num_samples[num_samples.values < min_samples].index)
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

def prepare_data_single_policy(
    case_data: pd.DataFrame,
    policy_data_prepped: pd.DataFrame,
    bins_list: List,
    policies: str,
    file_id: Union[str, None] = None,
    save_path: str = "./data/single_policy_bins/",
    save_data: bool = True,
    force_rerun: bool = False,
    pbar: bool = True,
    new_df: Union[None, pd.DataFrame] = None,
) -> pd.DataFrame:

    ### reload the dataframe from file if applicable
    if file_id is None and isinstance(policies, str):
        file_id = policies
    elif file_id is None and not isinstance(policies, str):
        raise ValueError("if passing multiple policies, you must pass a file id")

    filename = (
        file_id.replace(" - ", "_")
        + "-bins="
        + "".join([str(b[0]) + "-" + str(b[1]) + "_" for b in bins_list])[:-1]
        + ".csv"
    )

    if not force_rerun and os.path.exists(save_path + filename):
        new_df = pd.read_csv(save_path + filename, index_col=0, header=[0, 1])
        new_df[("info", "date")] = pd.to_datetime(
            new_df[("info", "date")], format="%Y-%m-%d"
        )
        return new_df

    ### initialize the new dataframe
    if new_df is None:
        new_df = prepare_new_df(case_data)

    # 3 possible cases for policies:
    # 1) None (use all policies)
    # 2) str (use this specific policy)
    # 3) List (use the given list of policies)

    if policies is None:
        policies = policy_data_prepped["full_policy"].unique()
    elif isinstance(policies, str):
        policies = [policies]

    tuples_policies = [
        (p, (str(date_range[0]) + "-" + str(date_range[1])))
        for date_range in bins_list
        for p in policies
    ]

    cols_polices = pd.MultiIndex.from_tuples(tuples_policies)
    policies_df = pd.DataFrame(columns=cols_polices)
    new_df = pd.concat([new_df, policies_df])
    new_df = new_df.fillna(0)
    policy_data_filtered = policy_data_prepped[
        policy_data_prepped["full_policy"].isin(policies)
    ]

    # generate dataframe
    df_dict = policy_data_filtered.to_dict("records")

    for row in tqdm(df_dict, disable=not pbar):
        for date_bin in bins_list:
            for policy_name in policies:
                date_range = get_date_range(row["date"], date_bin[0], date_bin[1])

                # Generate label (this is the 2nd level label in the multiIndexed column)
                label = str(date_bin[0]) + "-" + str(date_bin[1])
                new_df.loc[
                    (new_df[("info", "date")].isin(date_range))
                    & (
                        (new_df[("info", "county")] == row["county"])
                        | (row["policy_level"] == "state")
                    )
                    & (new_df[("info", "state")] == row["state"]),
                    (policy_name, label),
                ] = 1

    if save_data:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        new_df.to_csv(save_path + filename)
    return new_df

def get_date_range(date, start_move=0, stop_move=7):
    """Get the date range from date+start_move to date+stop_move"""

    return pd.date_range(
        start=date + timedelta(days=start_move), end=date + timedelta(days=stop_move)
    )

def get_single_policy_regression_data(
                    policy_name,
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
    
def fit_ols_model_single_policy(data,
                                policy_name,
                                dep_var,
                                use_const=True):
    """Fit an ols model from statsmodels

    Parameters
    ----------
    data

    policy_name

    dep_var

    use_const

    Returns
    ---------
    dictionary containing the coefficience (params), standard error of the coefficients (std_err),
    r^2 value (r_squared) and the p values (p_values)
    """
    y = data[('info', dep_var)]
    X = data[policy_name]

    if use_const:
        X = sm.add_constant(X)

    model = sm.OLS(y, X)
    results = model.fit()

    results_dict = {
        'r_squared': results.rsquared,
        'p_values': results.pvalues.to_dict(),
        'params': results.params.to_dict(),
        'std_err': results.bse.to_dict()
    }

    return results_dict
