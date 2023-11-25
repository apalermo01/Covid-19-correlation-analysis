import os
from datetime import timedelta

import numpy as np
import pandas as pd

from covid_project.data_utils import get_cases
from tqdm.notebook import tqdm
import us


def generate_state_case_dict(
    case_df, save_data=False, load_from_file=False, path="./data/state_data/"
):
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
        state_save_path = path + str(state) + ".csv"

        if load_from_file and os.path.exists(state_save_path):
            state_data = pd.read_csv(state_save_path, index_col=0)
        else:
            state_data = get_cases(case_dataframe=case_df, level="state", state=state)
            if save_data:
                if not os.path.exists(path):
                    os.makedirs(path)
                state_data.to_csv(state_save_path)
        state_cases_dict[state] = state_data
    return state_cases_dict


def calculate_deltas(
    case_df,
    policy_df,
    measure_period=14,
    filtered_policies=None,
    state_cases_dict=None,
    save_state_data=True,
    load_state_data_from_file=True,
    state_data_path="./data/state_data/",
    results_path="./data/deltas/",
    force_run=False,
    save_results=True,
    disable_pbar=False,
):
    """For every policy implementation at the state and county level, calculate the change in case and death numbers.

    Parameters
    ----------
    case_df : pandas DataFrame
        output of clean_case_data()
    policy_df : pandas DataFrame
        output of clean_policy_data()
    measure_period : int
        time to wait (in days) before measuring a change in new case or death numbers (14 by default)
    filtered_policies : array-like
        specify policies to select (default: None- calulate deltas for all policies)
    state_cases_dict :
    save_state_data :
    load_state_data_from_file :
    state_data_path :
    results_path :
    force_run :
    save_results :
    disable_pbar : boolean
        If true, suppresses progress bar


    Returns
    ----------
    A copy of the covid policies df with 2 appended columns for the change in case and death numbers.
    """

    # Load all state-aggregated datasets into a dictionary. We expect to need all 50 states so let's take the time to aggregate
    # the state data now so we don't need to do it repeatedly in the loop.

    if state_cases_dict is None:
        state_cases_dict = generate_state_case_dict(
            case_df=case_df,
            save_data=save_state_data,
            load_from_file=load_state_data_from_file,
            path=state_data_path,
        )

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

        start = ser[ser.index == date].values[0]
        start_1day = ser[ser.index == date + day_1].values[0]

        end = ser[ser.index == date + wait].values[0]
        end_1day = ser[ser.index == date + wait + day_1].values[0]

        return [start, start_1day, end, end_1day]

    # If there we are only examining select policies, then filter those out.
    if filtered_policies is not None:
        policy_df = policy_df.loc[policy_df["policy_type"].isin(filtered_policies)]

    correlated_df = policy_df.copy()

    # Initially fill the delta column with nan.
    correlated_df.loc[:, f"case_{measure_period}_day_delta"] = np.nan
    correlated_df.loc[:, f"case_{measure_period}_day_accel"] = np.nan
    correlated_df.loc[:, f"death_{measure_period}_day_delta"] = np.nan
    correlated_df.loc[:, f"death_{measure_period}_day_accel"] = np.nan

    # case_df['date'] = datetime.strptime(case_df['date'].str, "%Y-%m-%d")
    case_df["date"] = pd.to_datetime(case_df["date"], format="%Y-%m-%d")
    case_df = case_df.set_index("date")
    total_policies = len(policy_df)

    for index, data in tqdm(policy_df.iterrows(), disable=disable_pbar):
        # print(data['date'])
        # data['date'] = datetime.strptime(data['date'], "%Y-%m-%d")

        # If this is a state-level policy, then we already have the DataFrame to use.
        if data.policy_level == "state":
            state_df = state_cases_dict[data.state]
            ser_cases = state_df["new_cases_7day_1e6"]
            ser_deaths = state_df["new_deaths_7day_1e6"]

        # This must be a county level policy- filter the appropriate data.
        else:
            ser_cases = case_df["new_cases_7day_1e6"][
                case_df["fips_code"] == data.fips_code
            ]
            ser_deaths = case_df["new_deaths_7day_1e6"][
                case_df["fips_code"] == data.fips_code
            ]

        # Get the case and death numbers at the appropriate days.
        c11, c12, c21, c22 = sub_calc_deltas(ser_cases, date=data.date)
        d11, d12, d21, d22 = sub_calc_deltas(ser_deaths, date=data.date)

        # Calculate the difference in new cases at the selected dates.
        correlated_df.at[index, f"case_{measure_period}_day_delta"] = c21 - c11
        correlated_df.at[index, f"death_{measure_period}_day_delta"] = d21 - d11

        # Calculate the change in curvature (aka acceleration) of the case / death plots at policy implementation and
        # measure_period days afterwards.

        correlated_df.at[index, f"case_{measure_period}_day_accel"] = (
            (c12 - c11) - (c21 - c22)
        ) / measure_period
        correlated_df.at[index, f"death_{measure_period}_day_accel"] = (
            (d12 - d11) - (d21 - d22)
        ) / measure_period

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
    policy_types = [elem + " - start" for elem in deltas["policy_type"].unique()] + [
        elem + " - stop" for elem in deltas["policy_type"].unique()
    ]

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
        case_data = deltas[
            (deltas["policy_type"] == policy[:len_index])
            & (deltas["start_stop"] == start_stop)
        ][f"case_{measure_period}_day_delta"]

        death_data = deltas[
            (deltas["policy_type"] == policy[:len_index])
            & (deltas["start_stop"] == start_stop)
        ][f"death_{measure_period}_day_delta"]

        case_accel_data = deltas[
            (deltas["policy_type"] == policy[:len_index])
            & (deltas["start_stop"] == start_stop)
        ][f"case_{measure_period}_day_accel"]

        death_accel_data = deltas[
            (deltas["policy_type"] == policy[:len_index])
            & (deltas["start_stop"] == start_stop)
        ][f"death_{measure_period}_day_accel"]

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
    delta_stats = pd.DataFrame(
        np.transpose(
            [
                case_avg,
                case_accel_avg,
                death_avg,
                death_accel_avg,
                case_std,
                case_accel_std,
                death_std,
                death_accel_std,
                num_samples,
            ]
        ),
        index=policy_types,
        columns=[
            "case_avg",
            "case_accel_avg",
            "death_avg",
            "death_accel_avg",
            "case_std",
            "case_accel_std",
            "death_std",
            "death_accel_std",
            "num_samples",
        ],
    )

    # Drop record with less than min_samples samples.
    delta_stats.drop(
        delta_stats[delta_stats["num_samples"] <= min_samples].index, inplace=True
    )

    return delta_stats
