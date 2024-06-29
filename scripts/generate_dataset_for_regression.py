from covid_project.generate_dataset_group import generate_dataset_group
from covid_project.policy_mappings import 

if __name__ == '__main__':
    all_bins = [
        [(0, 7), (8, 999)],
        [(0, 14), (15, 999)],
        [(0, 20), (21, 999)],
        [(0, 7), (8, 14), (15,999)],
        [(0, 7), (8, 21), (22, 999)],
        [(0, 7), (8, 14), (15, 28), (29, 60), (61, 999)]

    for bins_list in all_bins:
        generate_dataset_group(bins_list, policy_dict, min_samples=3)
