"""
collection of functions to run and process regression analysis
"""


from covid_project.data_utils import clean_covid_data, clean_policy_data, prep_policy_data
from covid_project.policy_mappings import policy_dict_v1
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

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

    cols = ['dep_var', 'policy', 'bins_list', 'bin', 'r_squared', 'p_value', 'param', 'std_err']
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
                          sort_values = True,
                          ax = None):
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
    
    ### data preparation
    data = data[(data['dep_var']==dep_var)]
    data = data[['policy', 'bins_list', 'rsquared']]
    data = data.drop_duplicates()
    data = data.set_index('policy')
    data = data.pivot(columns='bins_list')
    
    ### rename columns
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
    
    ### optionally sort the dataframe such that the
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
    return ax, bins_ids