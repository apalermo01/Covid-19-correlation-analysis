"""
collection of functions to run and process regression analysis
"""


import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from covid_project.data_utils import (clean_covid_data, clean_policy_data,
                                      prep_policy_data)
from covid_project.policy_mappings import policy_dict_v1




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
###########################################
# Functions from the old Covid Data Class #
###########################################


def prep_policy_data(policy_data,
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
    for key in policy_dict_v1.keys():
        policy_data['policy_type'].replace(
            to_replace=key, value=policy_dict_v1[key], inplace=True)

    # Define a new field that includes policy_type, start_stop,
    # and policy_level information
    policy_data.loc[:, 'full_policy'] =\
    policy_data['policy_type'] + " - " +\
    policy_data['start_stop'] + " - " +\
    policy_data['policy_level']

    # Get number of times each policy was implemented.
    num_samples = proc_policy_data['full_policy'].value_counts()

    # drop the policy if it was implemented fewer than min_policy times.
    proc_policy_data = proc_policy_data.drop(proc_policy_data[
        proc_policy_data['full_policy'].isin(
                num_samples[num_samples.values < min_samples].index)
    ].index)

    # return the DataFrame
    return proc_policy_data


def join_policies(case_df,
                  policy_df,
                  output=True,
                  bins_list=[(0, 6), (7, 13), (14, 999)],
                  state_output=False,
                  save_csv=True,
                  load_csv=True,
                  cutoff_date=pd.to_datetime("2020-12-31"),
                  filter_counties=None):
    """Data preprocessing for Machine Learning that joins case and policy data.

    This function works by generating a processed DataFrame for each state and
    the concatenating the resulting 50 DataFrames at the very end.

    Parameters
    ------------
    case_df: DataFrame
            case data
    policy_df: DataFrame
            policy data
    output: boolean
            output time elapsed after running
    bins_list: list
            list of tuples with the desired date ranges
            default: [(0, 6), (7, 13), (14, 999)]
    state_output: boolean
            output time elapsed for filtering with each state
            default: False

    Returns
    ------------
    df3: DataFrame
            Processed DataFrame
    """
    time_start = time.time()
    filename = cfg.DATASET_DIR + f"prepped_data_{str(bins_list)}.csv"

    if load_csv and os.path.exists(filename):
        df3 = pd.read_csv(filename, header=[0, 1], index_col=[0])
        print("loaded data from csv")
        return df3

    # Define a sub-function to convert passed integers to the desired
    # date range starting from a given date.
    def _get_date_range(date, start_move=0, stop_move=7):

        # Get the date range from date+start_move to date+stop_move
        return pd.date_range(start=date+timedelta(days=start_move),
                             end=date+timedelta(days=stop_move))

    # Ensure all date columns are datetime
    case_df.loc[:, 'date'] = pd.to_datetime(
        case_df.loc[:, 'date'], format='%Y-%m-%d')
    policy_df.loc[:, 'date'] = pd.to_datetime(
        policy_df.loc[:, 'date'], format='%Y-%m-%d')

    # filter dates after the cutoff
    case_df = case_df[case_df['date'] < cutoff_date]
    policy_df = policy_df[policy_df['date'] < cutoff_date]

    # filter to specific counties
    if filter_counties is not None:
        case_df = case_df[case_df['full_loc_name'].isin(filter_counties)]

    # Make list of all policies.
    all_policies = policy_df['full_policy'].unique()

    # Construct multiIndex for df3.
    tuples_info = [("info", "state"), ("info", "county"), ("info", "full_loc"),
                   ("info", "date"), ("info", "new_cases_1e6")]

    tuples_policies = [(policy,
                        (str(date_range[0]) + "-" + str(date_range[1])))
                       for policy in all_policies for date_range in bins_list]

    tuples_index = tuples_info + tuples_policies
    col_index = pd.MultiIndex.from_tuples(tuples_index)

    # Get list of all states.
    all_states = case_df['state'].unique()

    # Generate a list of empty dataframes- one for each state.
    frames = [pd.DataFrame() for state in all_states]

    # Loop through all states.
    for (i, state) in enumerate(all_states):

        if output:
            state_time_start = time.time()

        # Initiallize dataFrame.
        frames[i] = pd.DataFrame(columns=pd.MultiIndex.from_tuples(col_index),
                                 data={
            ("info", "state"): state,
                ("info", "county"): case_df['county'][case_df['state'] == state],
                ("info", "full_loc"): case_df['full_loc_name'][
                    case_df['state'] == state
                    ],
                ("info", "date"): case_df['date'][case_df['state'] == state],
                ("info", "new_cases_1e6"): case_df['new_cases_1e6'][
                    case_df['state'] == state]
        })

        # Filter policy data to only those that were enacted in that state.
        filtered_policy = policy_df[policy_df['state'] == state]

        # Loop through every policy that was implemented in the current state.
        for (date, county, policy, level) in zip(filtered_policy['date'],
                                                 filtered_policy['county'],
                                                 filtered_policy[
            'full_policy'],
                                                 filtered_policy[
            'policy_level']):
            # Loop through each bin
            for date_bin in bins_list:

                # calculate the date range
                date_range = _get_date_range(date, date_bin[0], date_bin[1])

                # Generate label (this is the 2nd level label in the
                # multiIndexed column)
                label = (str(date_bin[0]) + "-" + str(date_bin[1]))

                # For every record in frames that falls within the date range
                # of the specific policy, set the appropriate column to 1.
                frames[i].loc[(
                    (frames[i]['info', 'date'].isin(date_range)) &
                    ((frames[i]['info', 'county'] == county) |
                         (level == 'state')) &
                    (frames[i]['info', 'state'] == state)),
                    (policy, label)] = 1

        if state_output:
            state_time_end = time.time()
            print(f"{state}: {state_time_end - state_time_start} seconds")

    # Concat the DataFrames
    df3 = pd.concat([frames[i] for i in range(len(frames))])

    # Fill NaNs
    df3.fillna(0, inplace=True)
    time_end = time.time()

    if save_csv:
        df3.to_csv(filename)
        print("csv saved")
    if output:
        print("data shaped\nbins: {}\ntime elapsed: {}".format(
            bins_list, (time_end - time_start)))

    return df3


def county_split(df, test_size):
    """Do a train-test split based on the county.
    """

    all_counties = df['full_loc_name'].unique()

    # shuffle the list
    np.random.shuffle(all_counties)

    # split the data
    counties_test = all_counties[: int(len(all_counties)*test_size)]
    counties_train = all_counties[int(len(all_counties)*test_size):]

    df_test = df[df['full_loc_name'].isin(counties_test)]
    df_train = df[df['full_loc_name'].isin(counties_train)]

    return df_test, df_train


def train_single_model(self,
                       df_train_proc,
                       model_in,
                       metrics_dict,
                       K=10,
                       verbose=True,
                       save_output=False,
                       filename="log.txt"):
    """Function to train models using K-fold cross validation
    Parameters
    -----------
    df_train_proc: DataFrame
            processed dataframe (after selecting bins and joining with cases)
    model_in: call to an sklearn object
            call to the models constructor method
    K: integer
            number of cross-validation folds
    verbose: Boolean
            detailed outputs

    """

    results_dict = {metric: [] for metric in metrics_dict.keys()}

    # get a list of all unique counties
    counties = df_train_proc[('info', 'full_loc')].unique()

    # shuffle the counties
    np.random.shuffle(counties)

    # declare batch size
    batch_size = int(len(counties) / K)

    # logging
    msg1 = f"number of cross-validation folds: {K}"
    msg2 = f"num counties in validation set: {batch_size}"

    if verbose:
        print(msg1)
        print(msg2)
    if save_output:
        with open(filename, "a") as log:
            log.write(msg1)
            log.write(msg2)

    # loop through cross validation folds
    for k in range(K):

        # select the train and validation portion
        df_train = df_train_proc[~df_train_proc[
            ('info', 'full_loc')].isin(counties[k*batch_size:(k+1)*batch_size])]

        df_validate = df_train_proc[df_train_proc[
            ('info', 'full_loc')].isin(counties[k*batch_size:(k+1)*batch_size])]

        # Implement and train the model
        X_train = df_train.loc[:, df_train.columns[5:]].values
        y_train = df_train.loc[:, ('info', 'new_cases_1e6')].values

        X_validate = df_validate.loc[:, df_validate.columns[5:]].values
        y_validate = df_validate.loc[:, ('info', 'new_cases_1e6')].values

        model = model_in
        model.fit(X_train, y_train)

        # compute scores
        for metric in metrics_dict.keys():
            score = metrics_dict[metric](y_validate, model.predict(X_validate))
            results_dict[metric].append(score)

        results = [(str(metric) + ": " + str(results_dict[metric][k]))
                    for metric in metrics_dict.keys()]

        msg = f"fold: {k}, scores: {results}"
        if verbose:
            print(msg)
        if save_output:
            with open(filename, "a") as log:
                log.write(msg)

    return results_dict, model.get_params()


def loop_models(self,
                df_train_proc,
                models_dict,
                metrics_dict,
                bin_id,
                K=10,
                verbose=True,
                save_output=False,
                filename="log.txt",):
    """Trains a series of models

    """
    # Declare empty dict to hold all results
    results = {}

    # loop through all models passed
    for model in models_dict.keys():
         msg = f"running model: {model}"

         if verbose:
             print(msg)
         if save_output:
             with open(filename, "a") as log:
                 log.write(msg)

         # declare emtpy dictionary for results for a single run
         model_results = {}
         scores, params = train_single_model(df_train_proc=df_train_proc,
                                             model_in=models_dict[model],
                                             metrics_dict=metrics_dict,
                                             K=K,
                                             verbose=verbose,
                                             save_output=save_output,
                                             filename=filename,)

         # save the results in a dictionary
         model_results['params'] = params
         model_results['scores'] = scores

         results[model] = model_results

    return results


def run_batch(self,
              bins_dict,
              models_dict,
              metrics_dict,
              case_data=None,
              policies=None,
              K=10,
              verbose=True,
              save_output=False,
              filename=None,
              overwrite=True,
              output_json=False,
              json_file=None,):
    results = {}

    if case_data is None:
        case_data = self.data
    if policies is None:
        policies = self.prepped_policy_data
    if overwrite & os.path.exists(filename):
        os.remove(filename)

    for i, key in enumerate(bins_dict):
        bins_list = bins_dict[key]

        msg = f"bins: {bins_list}"
        if verbose:
            print(msg)
        if save_output:
            with open(filename, "a") as log:
                log.write(msg)
        try:
            df_train_proc = self.join_policies(case_df=case_data,
                                               policy_df=policies,
                                               output=True,
                                               bins_list=bins_list,
                                               state_output=False,
                                               save_csv=True,
                                               load_csv=True,
                                               cutoff_date=pd.to_datetime("2020-12-31"),
                                               filter_counties=self.train_counties,
                                               )

            models_results = self.loop_models(df_train_proc=df_train_proc,
                                              models_dict=models_dict,
                                              metrics_dict=metrics_dict,
                                              K=K,
                                              bin_id=str(bins_list),
                                              verbose=verbose,
                                              save_output=save_output,
                                              filename=filename)
        except Exception as e:
            print("EXCEPTION THROWN: ", e)
            continue
        models_results['bins'] = bins_list
        results[key] = models_results

        if output_json:
            if overwrite & os.path.exists(json_file):
                os.remove(json_file)

            with open(json_file, 'w') as file:
                print("saving json")
                json.dump(results, file, sort_keys=True, indent=4)

    return results
