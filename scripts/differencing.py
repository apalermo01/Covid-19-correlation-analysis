"""
Generate first and second-order differences on a range of different dates.

This script is generating the datasets used in notebook 04
"""

import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
from covid_project import data_cleaning as dc
from covid_project.diffs import calculate_diffs, generate_state_case_dict
# from tqdm import tqdm
from tqdm.auto import tqdm
import warnings
import matplotlib.pyplot as plt

# def main():
#     from time import sleep
#     for i in tqdm(range(4, 30)):
#        sleep(0.1) 
def main():
    case_data = dc.clean_covid_data()
    policy_data = dc.clean_policy_data()
    state_case_dict = generate_state_case_dict(case_data,
                                               save_data = True,
                                               load_from_file = True)
    for measure_period in range(4, 30):
        _, _ = calculate_diffs(
                case_df = case_data,
                policy_df = policy_data,
                measure_period = measure_period,
                state_cases_dict = state_case_dict)

if __name__ == '__main__':
    main()
