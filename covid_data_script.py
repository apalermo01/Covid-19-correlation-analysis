# imports

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import us
import requests
import policy_dict
import time


def retrieve_data(load_local):
    """Load the case and policy data."""

    if load_local:
        df = pd.read_csv("case_data_clean.csv", index_col=0)
        df2 = pd.read_csv("policy_data_clean.csv", index_col=0)

        # Convert date fields to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')

    else:
        df = pd.read_csv(
            'https://query.data.world/s/jbgdegbanosfmgly7etz2gxqsbhflk'
            )
        html = requests.get(
            ("https://catalog.data.gov/dataset/"
             "covid-19-state-and-county-policy-orders")
        ).text

        policy_file = html.split(
            "a href=\"/dataset/covid-19-state-and-county-policy-orders")[1]\
            .split("<span>plotly</span>")[0]\
            .split("https://plot.ly/external/?url=")[1]\
            .split("\">")[0]
        df2 = pd.read_csv(policy_file)

    return df, df2


def clean_data(df, df2):
    """Clean the case and policy data."""

    # Cleaning df ############################################################

    # Rename columns.
    df.rename(columns={
        'cumulative_cases_per_100_000': 'cumulative_cases_1e6',
        'cumulative_deaths_per_100_000': 'cumulative_deaths_1e6',
        'new_cases_per_100_000': 'new_cases_1e6',
        'new_deaths_per_100_000': 'new_deaths_1e6',
        'new_cases_7_day_rolling_avg': 'new_cases_7day',
        'new_deaths_7_day_rolling_avg': 'new_deaths_7day',
        'location_name': 'county'
    }, inplace=True)

    # Drop location_types that are not counties.
    df = df.drop(df[df['location_type'] != 'county'].index)

    # Get list of u.s. states
    states = [elem.name for elem in us.states.STATES]

    # Drop anything not in one of the 50 states (also drops DC).
    df = df.drop(df[~df['state'].isin(states)].index)

    # Generate a new column with the combined state and county name.
    df['full_loc_name'] = df['location_name'] + ', ' + df['state']

    # Make all counties lowercase for consistency.
    df['location_name'] = df['location_name'].str.lower()

    # Re-order columns.
    cols = df.columns.tolist()
    cols = cols[:6] + [cols[-1]] + cols[6:-1]
    df = df[cols]

    # Rename location name to county
    # df.rename(columns={'location_name': 'county'}, inplace=True)

    # Cast dates to datetime.
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # In numerical columns, fill na values with zero and change any negative
    # values to positive.
    cols = ['new_cases', 'new_deaths', 'new_cases_1e6', 'new_deaths_1e6']

    for col in cols:
        df[col].fillna(value=0, inplace=True)
        df[col].clip(lower=0, inplace=True)

    # Case all numerical columns to integers.
    cols2 = ["fips_code", "total_population", "new_cases", "new_deaths"]
    df[cols2] = df[cols2].astype(np.int64)

    # Set rolling averages to zero in the first few days of measurements.
    df[['new_cases_7day', 'new_deaths_7day']] = df[['new_cases_7day',
                                                    'new_deaths_7day']].mask(

        (df['date'] < pd.to_datetime('2020-01-30', format='%Y-%m-%d')),
        df[['new_cases_7day', 'new_deaths_7day']].fillna(0)
        )

    # Re-calculate 7-day rolling average to replace nulls ####################

    # Get datapoints where normalized new_cases and new_deaths are null
    nulls_case7day = df[df['new_cases_7day'].isnull()]
    nulls_death7day = df[df['new_deaths_7day'].isnull()]

    # Pre-define timedelta for efficiency.
    days_7 = timedelta(days=7)

    # Loop through nulls in new_cases_7day
    for index, data in nulls_case7day.iterrows():

        # Calculate rolling average at the datapoint of interest
        df.loc[index, ['new_cases_7day']] = np.sum((
            [df['new_cases'][(df['full_loc_name'] == data.full_loc_name) &
                             ((df['date'] <= data.date) &
                              (df['date'] > data.date-days_7))
                             ].values])) / 7

    # Loop through nulls in new_deaths_7day
    for index, data in nulls_death7day.iterrows():

        # Calculate rolling average at datapoint of interest.
        df.loc[index, ['new_deaths_7day']] = np.sum(([df['new_deaths'][
            (df['full_loc_name'] == data.full_loc_name) &
            ((df['date'] <= data.date) &
             (df['date'] > data.date-days_7))].values])) / 7

    # Normalize 7-day rolling averages against population.
    df["new_cases_7day_1e6"] = df['new_cases_7day'] /\
        (df['total_population'] / 1e5)

    df["new_deaths_7day_1e6"] = df['new_deaths_7day'] /\
        (df['total_population'] / 1e5)

    # Cleaning df2 ###########################################################

    # get abbreviations for each state
    abbr = [elem.abbr for elem in us.states.STATES]

    # Exclude territories and DC
    df2 = df2.drop(df2[~df2['state_id'].isin(abbr)].index)

    # Replace state names with abbreviations
    df2.replace(to_replace=us.states.mapping('abbr', 'name'), inplace=True)

    # Rename 'state_id' column
    df2.rename(columns={'state_id': 'state'}, inplace=True)

    # Drop 'total phases' since it's not useful
    df2.drop('total_phases', axis=1, inplace=True)

    # If 'county' is null, then it's a statewide policy, so repalce it with
    # 'statewide' and convert to lowercase for consistency
    df2['county'].fillna(value='statewide', inplace=True)
    df2['county'] = df2['county'].str.lower()

    # Fix some formatting issues in 'date': 0020 instead of 2020, then
    # convert to datetime
    df2.loc[:, 'date'] = '2020' + df2['date'].str[4:]
    df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')

    # Fix some mismatches between county names
    # (e.g. 'bronx' in df is 'bronx county' in df2)
    county_match = re.compile(" county$")
    munici_match = re.compile(" municipality$")
    Borough_match = re.compile(" borough$")

    df2['county'].replace(to_replace=county_match,  value='', inplace=True)
    df2['county'].replace(to_replace=munici_match,  value='', inplace=True)
    df2['county'].replace(to_replace=Borough_match, value='', inplace=True)

    # Update fips codes for states
    for index, data in df2.iterrows():
        if data.policy_level == 'state':
            df2.loc[index, 'fips_code'] =\
                np.int64(us.states.lookup(data.state).fips)

    df2['fips_code'] = df2['fips_code'].astype(np.int64)

    # Drop uninformative policies
    policies_drop = ['New Phase', 'Phase 1', 'Phase 2', 'Phase 2, 3',
                     'Phase 3', 'Phase 3 Step 2', 'Phase 4', 'Phase 5']

    df2 = df2.drop(df2[df2['policy_type'].isin(policies_drop)].index)

    # Rename policies that have unreasonably long names
    policy_replacements_dict = {
        'Stop Initiation Of Evictions Overall Or Due To Covid Related Issues':
            'Stop Initiation Of Evictions',

        'Modify Medicaid Requirements With 1135 Waivers Date Of CMS Approval':
            'Modify Medicaid Requirements',

        'Stop Enforcement Of Evictions Overall Or Due To Covid Related Issues':
            'Stop Enforcement Of Evictions',

        'Mandate Face Mask Use By All Individuals In Public Facing Businesses':
            'Mandate Face Masks In Businesses',

        'Mandate Face Mask Use By All Individuals In Public Spaces':
            'Mandate Face Masks In Public Spaces',

        'Reopened ACA Enrollment Using a Special Enrollment Period':
            'ACA Special Enrollment Period',

        'Suspended Elective Medical Dental Procedures':
            'Suspend Elective Dental Procedures',

        'Allow Expand Medicaid Telehealth Coverage':
            'Expand Medicaid Telehealth Coverage',

        'Renter Grace Period Or Use Of Security Deposit To Pay Rent':
            'Grace Period / Security Deposit for Rent'
    }

    for key in policy_replacements_dict.keys():
        df2['policy_type'].replace(
            to_replace=key, value=policy_replacements_dict[key], inplace=True
            )

    df2['policy_type'] = df2['policy_type'].str.lower()

    # Drop policies implemented before the measure period or
    # planned for the future.
    df2 = df2.drop(df2[(df2['date'] < min(df['date'])) |
                       (df2['date'] > datetime.today())].index)

    return df, df2


def prep_policy_data(df2,
                     policy_dict=policy_dict.policy_dict,
                     min_samples=10):
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

    proc_policy_data = df2.copy()

    # Replace policies with the ones in policy_dict().
    for key in policy_dict.keys():
        proc_policy_data['policy_type'].replace(
            to_replace=key, value=policy_dict[key], inplace=True)

    # Define a new field that includes policy_type, start_stop,
    # and policy_level information
    proc_policy_data.loc[:, 'full_policy'] =\
        proc_policy_data['policy_type'] + " - " +\
        proc_policy_data['start_stop'] + " - " +\
        proc_policy_data['policy_level']

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
                  state_output=False):
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

    # Define a sub-function to convert passed integers to the desired
    # date range starting from a given date.
    def _get_date_range(date, start_move=0, stop_move=7):

        # Get the date range from date+start_move to date+stop_move
        return pd.date_range(start=date+timedelta(days=start_move),
                             end=date+timedelta(days=stop_move))

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

    if output:
        print("data shaped\nbins: {}\ntime elapsed: {}".format(
            bins_list, (time_end - time_start)))

    return df3


if __name__ == "__main__":
    print("covid data script loaded")
