from covid_project.data_utils import generate_dataset_group
from covid_project.pol


if __name__ == '__main__':
    all_bins = [
        [(0, 14), (15, 999)],
        [(0, 14), (15, 28), (29, 999)],
        [(0, 7), (8, 14), (15, 999)],
        [(0, 7), (8, 14), (15, 28), (29, 60), (61, 999)]
    ]

    for bins_list in all_bins:
        generate_dataset_group(bins_list, policy_dict, min_samples=3)
