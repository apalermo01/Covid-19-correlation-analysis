"""
Functions for notebooks 5 and 6 - linear regression
"""
def generate_dataset_group(bins_list,
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
            policy_name = policy,
            bins_list = bins_list,
            pbar = False,
            new_df=new_df)


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
