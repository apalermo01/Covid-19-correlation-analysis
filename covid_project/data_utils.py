import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union
from covid_project.data_cleaning import clean_covid_data, clean_policy_data

import numpy as np
import pandas as pd
import us
from sodapy import Socrata
from tabulate import tabulate
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def get_cases(case_dataframe=None, level="county", county="orange", state="California"):
    """Return the new_case and new_death numbers at the given level of aggregation (county, state, or national).

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
        return_case_dataframe = case_dataframe[
            (case_dataframe["county"] == county) & (case_dataframe["state"] == state)
        ].set_index("date")[
            [
                "new_cases_1e6",
                "new_deaths_1e6",
                "new_cases_7day_1e6",
                "new_deaths_7day_1e6",
            ]
        ]
        return return_case_dataframe

    # If this is filtered at the state level, filter df to desired state. Otherwise, return national-level data.
    if level == "state":
        case_dataframe = case_dataframe[case_dataframe["state"] == state]

    # Reindex on location name.
    case_dataframe = case_dataframe.set_index(["full_loc_name"])

    # Get a list of all dates.
    all_dates = case_dataframe["date"].unique()

    # Get the total population from the county populations.
    total_population = sum(
        [
            (pops / 1e5)
            for pops in case_dataframe[case_dataframe["date"] == all_dates[0]][
                "total_population"
            ]
        ]
    )

    # Add up the case and death #s that have the same date.
    new_cases = [
        sum(
            [
                (county_cases / total_population)
                for county_cases in case_dataframe[case_dataframe["date"] == dates][
                    "new_cases_1e6"
                ]
            ]
        )
        for dates in all_dates
    ]

    new_deaths = [
        sum(
            [
                (county_cases / total_population)
                for county_cases in case_dataframe[case_dataframe["date"] == dates][
                    "new_deaths_1e6"
                ]
            ]
        )
        for dates in all_dates
    ]

    new_cases_7day = [
        sum(
            [
                (county_cases / total_population)
                for county_cases in case_dataframe[case_dataframe["date"] == dates][
                    "new_cases_7day_1e6"
                ]
            ]
        )
        for dates in all_dates
    ]

    new_deaths_7day = [
        sum(
            [
                (county_cases / total_population)
                for county_cases in case_dataframe[case_dataframe["date"] == dates][
                    "new_deaths_7day_1e6"
                ]
            ]
        )
        for dates in all_dates
    ]

    return_case_dataframe = pd.DataFrame(
        data={
            "date": all_dates,
            "new_cases_1e6": new_cases,
            "new_deaths_1e6": new_deaths,
            "new_cases_7day_1e6": new_cases_7day,
            "new_deaths_7day_1e6": new_deaths_7day,
        }
    ).set_index(["date"])
    return return_case_dataframe


def get_policies(
    policy_dataframe=None,
    state="California",
    county="statewide",
    state_policies=True,
    county_policies=True,
):
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
        return policy_dataframe[
            (policy_dataframe["state"] == state)
            & (
                (policy_dataframe["county"] == county)
                | (policy_dataframe["county"] == "statewide")
            )
        ]

    # state policies only
    elif state_policies and not county_policies:
        return policy_dataframe[
            (policy_dataframe["state"] == state)
            & (policy_dataframe["county"] == "statewide")
        ]

    # county policies only
    else:
        return policy_dataframe[
            (policy_dataframe["state"] == state)
            & (policy_dataframe["county"] == county)
        ]

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
