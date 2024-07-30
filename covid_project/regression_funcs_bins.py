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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('future.no_silent_downcasting', True)

BINS = [
        [(0, 7), (8, 999)],
        [(0, 14), (15, 999)],
        [(0, 20), (21, 999)],
        [(0, 7), (8, 14), (15,999)],
        [(0, 7), (8, 21), (22, 999)],
        [(0, 7), (8, 14), (15, 28), (29, 60), (61, 999)]
    ]
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

def calculate_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def fit_ols_model_single_policy(data,
                                policy_name,
                                dep_var,
                                use_const=True,
                                ret_diagnostics=False):
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
    model = model.fit()

    results_dict = {
        'r_squared': model.rsquared,
        'p_values': model.pvalues.to_dict(),
        'params': model.params.to_dict(),
        'std_err': model.bse.to_dict()
    }

    if ret_diagnostics:
        pred = model.predict()
        resid = y - pred
        vif = calculate_vif(X)

        results_dict['predictions'] = pred
        results_dict['true'] = y
        results_dict['residuals'] = resid
        results_dict['vif'] = vif
        
    return results_dict
    
def collect_all_regression_results_to_df(regression_results_path):
    """Collect the results from every regression analysis found in the passed folder

    There should be a set of json files with the name format <dep_var>_bins=<bin1>_<bin2>....json

    Parameters
    ----------
    regression_results_path
        path to folder with results json files

    Returns
    ---------
    pandas dataframe with columns:
        dep_var, bins_list, policy, bin, rsquared, p_value, param, and std_err
    """
    results_files = os.listdir(regression_results_path)

    cols = ['dep_var', 'policy', 'bins_list', 'bin',
        'r_squared', 'p_value', 'param', 'std_err']
    all_data = dict()
    pk = 0
    for r in results_files:
        if 'bins' not in r:
            continue
        dep_var = r.split('bins')[0][:-1]
        bins_str = r.split('=')[1].split('.')[0]
        bins_list = [(int(e.split('-')[0]), int(e.split('-')[1]))
                     for e in bins_str.split('_')]
        with open(regression_results_path+r, "r") as f:
            data = json.load(f)

        for policy in data:
            r2 = data[policy]['r_squared']

            for b in data[policy]['p_values']:
                all_data[pk] = {
                    'dep_var': dep_var,
                    'bins_list': str(bins_list),
                    'policy': policy,
                    'bin': b,
                    'rsquared': r2,
                    'p_value': data[policy]['p_values'][b],
                    'param': data[policy]['params'][b],
                    'std_err': data[policy]['std_err'][b]
                }
                pk += 1

        df = pd.DataFrame.from_dict(all_data, orient='index')
    return df


def plot_rsquared_heatmap(data,
                          dep_var,
                          sort_values=True,
                          ax=None,
                          save_figure=False,
                          filename="./figures/rsquared_heatmap.png"):
    """Plots a heatmap of r-squared values for the given dependent variable. Generates a column
    for each set of bins and a row for each policy, where the color is the r-squared value.

    Parameters
    ----------
    data
        pandas dataframe: input data, return value of collect_all_regression_results_to_df

    dep_var
        string: dependent variable

    sort_values
        boolean: if true, sorts the plot such that the column containing the highest r-squared
        value appears in descending order

    ax:
        matlotlib axis handle

    Returns
    ---------
    ax
        axis handle containing the plot

    bin_ids
        dictionary of bins - used for reference on x-axis
    """

    # data preparation
    data = data[(data['dep_var'] == dep_var)]
    data = data[['policy', 'bins_list', 'rsquared']]
    data = data.drop_duplicates()
    data = data.set_index('policy')
    data = data.pivot(columns='bins_list')

    # rename columns
    # generate a set of bin ids
    # this cleans up the plot a bit
    # bin ids will also be returned for reference
    col_tuples = data.columns.values
    new_cols = []
    bins_ids = {}
    for i, col in enumerate(col_tuples):
        new_cols.append(f"bin_set_{i}")
        bins_ids[f"bin_set_{i}"] = col[1]

    data.columns = new_cols

    # optionally sort the dataframe such that the
    # bin containing the maximum r-squared appears sorted
    # in descending order
    if sort_values:
        maxes = data.idxmax().values
        max_vals = np.array([data.loc[m, :][i] for i, m in enumerate(maxes)])
        max_col = np.argmax(max_vals)
        data = data.sort_values(by=data.columns[max_col], ascending=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=[5, 10])

    ax = sns.heatmap(data, ax=ax)
    ax.set_title(f"r-squared results for " + dep_var)

    if save_figure:
        plt.savefig(filename)

    return ax, bins_ids