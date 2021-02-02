# imports

# basics
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

# visualizations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

# Linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# miscellaneous
import re
from tabulate import tabulate
from IPython.display import clear_output
import us
import requests



# load cleaned data (see 'Covid-19 data analysis.ipynb')
df = pd.read_csv("case_data_clean.csv")
df2 = pd.read_csv("policy_data_clean.csv")