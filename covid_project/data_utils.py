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

