"""
Collection of general plotting functions

Note that plotting functions related to the regression analysis is located in regression_funcs.py
"""

import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
import math
from covid_project.data_utils import clean_policy_data, get_cases, get_policies

"""
TODO: error mode in case and policy plot where bad county name is given
"""
# fontsizes for plots
BIG_FONT = 18
MED_FONT = 14
SMALL_FONT = 10





def plot_delta_stats(
    delta_stats,
    num_days=14,
    interval="std",
    save_figure=False,
    filename="eval delta stats figure.png",
):
    """Evaluate the correlations between policy implementations and new cases / deaths."""

    # force interval to std
    # interval = "std"
    assert interval in ['std', '0.9', '0.95', '0.99']
    fig, ax = plt.subplots(ncols=4, figsize=[10, 15], sharey=True)

    def eval_color(num, error):
        # green: policy seems to have a positive effect on cases
        # red: policy seems to have a negative effect on cases
        if (np.abs(num) - error > 0) and num < 0:
            return "g"
        elif (np.abs(num) - error > 0) and num > 0:
            return "r"
        else:
            return "k"

    for i, index in enumerate(delta_stats.index):
        vals = delta_stats.loc[index][:4].values
        vals_std = delta_stats.loc[index][4:8].values
        n = delta_stats.loc[index][8]

        for j, (col, val, val_std) in enumerate(zip(['case', 'case_accel', 'death', 'death_accel'], vals, vals_std)):
            # if interval is 'std', set the errorbar to the same width as the std
            if interval == "std":
                err = val_std
            # if a confidence interval is specified, set error bar to with width of the desired CI
            else:
                err = delta_stats[f'{col}_{interval}_ci_range'].loc[index]
            # print(f"val = {val}; error = {err}")
            ax[j].errorbar(
                y=i,
                x=val,
                xerr=err,
                marker=".",
                markersize=15,
                capsize=5,
                linewidth=3,
                linestyle="None",
                c=eval_color(val, err),
            )
    titles = [
        "new cases",
        "acceleration of new cases",
        "new deaths",
        "acceleration of new deaths",
    ]
    for i in range(4):
        lims = ax[i].get_ylim()
        ax[i].vlines(x=0, ymin=lims[0], ymax=lims[1], color="k")
        ax[i].set_ylim(lims)
        ax[i].set_title(titles[i])
        ax[i].tick_params(bottom=True, top=True, labelbottom=True, labeltop=True)
    plt.yticks(range(len(delta_stats.index)), delta_stats.index, fontsize=8)
    if interval == "std":
        title = f"Average change in covid metrics {num_days} days after implementation (errorbar = std)"
    else:
        title = f"Average change in covid metrics {num_days} days after implementation (CI = {interval})"
    plt.suptitle(title, y=0.95)

    if save_figure:
        plt.savefig(filename)

    return fig
