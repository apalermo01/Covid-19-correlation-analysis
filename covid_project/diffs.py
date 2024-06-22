
import os
from datetime import timedelta
from scipy import stats
import math
import numpy as np
import pandas as pd
from typing import List, Dict

from covid_project.data_utils import get_cases
from tqdm.notebook import tqdm
import us

def calculate_diffs(
        case_df: pd.DataFrame,
        policy_df: pd.DataFrame,
        measure_period: int = 14,
        filtered_policies: List = None,
        state_cases_dict: Dict = None,
        save_state_data: bool = True,
        load_state_data_from_file: bool = True,
        state_data_path: str = "./data/state_data/",
        results_path: str = "./data/diffs/",
        force_run: bool = False,
        save_results: bool = True,
        disable_pbar: bool = False,
):
    """For every policy implementation at the state and county level, calculate the change in case and death numbers.

    Parameters
    ----------
    case_df : DataFrame
        output of clean_case_data()
    policy_df : DataFrame
        output of clean_policy_data()
    measure_period : int
        time to wait (in days) before measuring a change in new case or death numbers (14 by default)
    filtered_policies : array-like
        specify policies to select (default: None- calulate differences for all policies)
    state_cases_dict : Dict
    save_state_data : bool
    load_state_data_from_file : bool
    state_data_path : str
    results_path : str
    force_run : bool
    save_results : bool
    disable_pbar : bool
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

    results_file = f"{results_path}{str(measure_period)}_diffs.csv"

    if os.path.exists(results_file) and not force_run:
        results = pd.read_csv(results_file, index_col=0)
        return results, state_cases_dict

    # Initialize wait period before measurement.
    wait_period = timedelta(days=measure_period)
    day_1 = timedelta(days=1)

    def sub_calc_diffs(ser, date, wait=wait_period):
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

    diff_df = policy_df.copy()

    # Initialize diff columns with nan
    diff_df.loc[:, f"case_{measure_period}_day_order_1"] = np.nan
    diff_df.loc[:, f"case_{measure_period}_day_order_2"] = np.nan
    diff_df.loc[:, f"death_{measure_period}_day_order_1"] = np.nan
    diff_df.loc[:, f"death_{measure_period}_day_order_2"] = np.nan

    # case_df['date'] = datetime.strptime(case_df['date'].str, "%Y-%m-%d")
    case_df["date"] = pd.to_datetime(case_df["date"], format="%Y-%m-%d")
    case_df = case_df.set_index("date")
    total_policies = len(policy_df)
    
    # go through every policy
    for index, data in tqdm(policy_df.iterrows(),
                            disable=disable_pbar,
                            total=len(policy_df),
                            desc=f"running differencing for time period {measure_period}"):

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
        c11, c12, c21, c22 = sub_calc_diffs(ser_cases, date=data.date)
        d11, d12, d21, d22 = sub_calc_diffs(ser_deaths, date=data.date)

        # Get first order differences
        diff_df.at[index, f"case_{measure_period}_day_order_1"] = c21 - c11
        diff_df.at[index, f"death_{measure_period}_day_order_1"] = d21 - d11

        # Calculate the change in curvature (aka acceleration) of the case / death plots at policy implementation and
        # measure_period days afterwards.
        diff_df.at[index, f"case_{measure_period}_day_order_2"] = (
            (c12 - c11) - (c21 - c22)
        ) / measure_period
        diff_df.at[index, f"death_{measure_period}_day_order_2"] = (
            (d12 - d11) - (d21 - d22)
        ) / measure_period

    if save_results:
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        diff_df.to_csv(results_file)
    return diff_df, state_cases_dict

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
