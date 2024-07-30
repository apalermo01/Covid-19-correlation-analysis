# Run this on first set up
from covid_project.data_utils import load_covid_data, load_policy_data
import os

def main():
    if 'data' not in os.listdir('./'):
        os.makedirs('./data/')
    load_covid_data("./data/covid_timeseries.csv", force_reload=False)
    load_policy_data("./data/covid_policies.csv", force_reload=False)

if __name__ == '__main__':
    main()
