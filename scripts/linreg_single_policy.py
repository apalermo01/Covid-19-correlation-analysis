"""
Generate datasets for notebook 5 - first version of regression
"""

from covid_project.regression_funcs_bins import generate_dataset_group_single_policy, fit_ols_model_single_policy, get_single_policy_regression_data
from covid_project.data_utils import get_all_policies
from covid_project.policy_mappings import policy_dict_v2
import os
from tqdm.auto import tqdm
import json
import argparse

all_bins = [
        [(0, 7), (8, 999)],
        [(0, 14), (15, 999)],
        [(0, 20), (21, 999)],
        [(0, 7), (8, 14), (15,999)],
        [(0, 7), (8, 21), (22, 999)],
        [(0, 7), (8, 14), (15, 28), (29, 60), (61, 999)]
    ]
def run_model_on_policies(bins,
                          all_policies,
                          dep_var,
                          pbar=True):
    """Loop to run the regression model on all policies"""
    
    results = dict()
    for policy in tqdm(all_policies, desc='running models'):
        suc, data = get_single_policy_regression_data(policy, bins)
        if not suc:
            print(f"[ERROR] data read failed: bins={bins}, policy={policy}, var={dep_var}")
            continue
        try:
            res = fit_ols_model_single_policy(data,
                                              policy,
                                              dep_var,
                                              True)
        except Exception as e:
            print(f"ERROR: {e}")
            print(f"policy: {policy}; bins: {bins}")
        results[policy] = res
    return results

def run_batch_of_models(all_bins,
                        all_policies,
                        dep_vars,
                        save_path="./data/regression_results_single_policy_bins/",
                        force_run = False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for bins_list in all_bins:
        for var in dep_vars:
            print(f"running models for {var} with bins {bins_list}")
            filename = var + "_bins=" + ''.join([str(b[0])+"-"+str(b[1])+"_" for b in bins_list])[:-1] + ".json"
            full_path = save_path + filename
            if os.path.exists(full_path) and not force_run:
                continue
            results = run_model_on_policies(bins=bins_list,
                                            all_policies=all_policies,
                                            dep_var=var,
                                            pbar=True)
            
            with open(full_path, "w") as f:
                json.dump(results, f, indent=2)

def generate_dataset():

    for bins_list in all_bins:
        print("bins: ", bins_list)
        generate_dataset_group_single_policy(bins_list, policy_dict_v2, min_samples=3)

def run_models():
    
    
    all_policies = get_all_policies(policy_dict = policy_dict_v2,
                                    min_samples = 3)
    
    dep_vars = [
        'new_cases_1e6',
        'new_deaths_1e6',
        'new_cases_7day_1e6',
        'new_deaths_7day_1e6',
    ]
    run_batch_of_models(all_bins=all_bins,
                    all_policies=all_policies,
                    dep_vars=dep_vars,)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate datasets or run regression models.')
    parser.add_argument('--run_what', choices=['generate_dataset', 'run_models'], help='Specify whether to generate datasets or run models.')
    args = parser.parse_args()

    if args.run_what == 'generate_dataset':
        generate_dataset()
    elif args.run_what == 'run_models':
        run_models()
    else:
        parser.print_help()