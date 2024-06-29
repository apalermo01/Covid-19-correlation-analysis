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





