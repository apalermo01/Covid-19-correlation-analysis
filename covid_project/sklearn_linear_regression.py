"""Functions for running linear regression through sklearn
on a single policy with a given set of bins"""

from covid_project.data_utils import (clean_covid_data, clean_policy_data,
                                      get_all_policies)
from covid_project.policy_mapping import policy_dict_v2


def get_processed_data(
    policy_name: str,
    bins_list: list[Tuple[int, int]],
    root_path="./data/single_policy_bins/",
) -> Tuple[bool, Union[None, pd.DataFrame]]:
    """Reads processed data. Generates the data if it doesn't already exist.
    Note: this is hard-coded to use policy_dict_v2 and 3 samples at minimum

    TODO: might be interesting to run some experiments with these conditions relaxed.

    Parameters
    ----------
    policy_name : str
        name of policy
    bins_list : list[Tuple[int, int]]
        list of bins to pull
    root_path : str, optional
        path to data folder, by default "./data/single_policy_bins/"

    Returns
    -------
    Tuple[bool, Union[None, pd.DataFrame]]
        First return is whether or not the pull was successful. False if the file was not found, True if it was.
        If True, returns the dataframe as the second argument, None otherwise
    """

    ### TODO: add option to generate processed data if the file is not present
    filename = (
        policy_name.replace(" - ", "_")
        + "-bins="
        + "".join([str(b[0]) + "-" + str(b[1]) + "_" for b in bins_list])[:-1]
        + ".csv"
    )
    path = root_path + filename

    if os.path.exists(path):
        data = pd.read_csv(root_path + filename, header=[0, 1], index_col=0)
        return True, data
    else:
        data = generate_dataset_group(bins_list, policy_dict_v2, min_samples=3)
        return False, None


def generate_dataset_group(bins_list, policy_dict, min_samples=3):
    """Generate datasets for every policy for a given group of bins
    Parameters
    ----------
    bins_list

    policy_dict

    min_samples
    """

    case_data = clean_covid_data()
    policy_data = clean_policy_data()

    policy_data_prepped = prep_policy_data(
        policy_data=policy_data, policy_dict=policy_dict, min_samples=min_samples
    )

    all_policies = policy_data_prepped["full_policy"].unique()
    new_df = prepare_new_df(case_data)

    for policy in tqdm(all_policies, desc="generating datasets for policies"):
        prepare_data_single_policy(
            case_data=case_data,
            policy_data_prepped=policy_data_prepped,
            policy_name=policy,
            bins_list=bins_list,
            pbar=False,
            new_df=new_df,
        )


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


def prepare_new_df(case_data):
    """Initialize the new dataframe"""

    dependent_vars = [
        "new_cases_1e6",
        "new_deaths_1e6",
        "new_cases_7day_1e6",
        "new_deaths_7day_1e6",
    ]

    tuples_info = [
        ("info", "location_type"),
        ("info", "state"),
        ("info", "county"),
        ("info", "date"),
    ]

    dependent_cols = [("info", e) for e in dependent_vars]
    tuples_info = tuples_info + dependent_cols
    case_data_cols = ["location_type", "state", "county", "date"]
    case_data_cols = case_data_cols + dependent_vars

    info_cols = pd.MultiIndex.from_tuples(tuples_info)
    new_df = pd.DataFrame(columns=info_cols)
    new_df[tuples_info] = case_data[case_data_cols]

    return new_df
