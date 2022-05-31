import matplotlib.pyplot as plt 
import seaborn as sns

# fontsizes for plots
BIG_TEXT   = 18
MED_TEXT   = 14
SMALL_TEXT = 10

def plot_counts_with_label(df, colname):
    """
    Generate a bar plot of the unique value counts for column <colname> of dataframe <df>
    :param df: pd.DataFrame - input data
    :param colname: column name to analyze
    :returns: axis handle
    """
    ax = df[colname].value_counts().plot.bar()
    for container in ax.containers:
        ax.bar_label(container)

    ax.set_title(f"counts for {colname}")
    ax.set_xlabel("attribute")
    ax.set_ylabel("count")

    return ax

def plot_nulls(df,
               title,
               subplots_kwargs={'figsize': [12, 5]}):
    """
    Visualize the number of nulls in a dataset

    Parameters
    ----------
    df

    title

    subplot_kwargs

    Returns 
    ---------
    """

    nulls = df.apply(lambda x: x.isnull().value_counts()).loc[True]

    fig, ax = plt.subplots(**subplots_kwargs)
    sns.set_style('darkgrid')
    sns.barplot(x=nulls.index, y=nulls.values, ax=ax, color='b')
    plt.xticks(rotation=60)
    ax.set_ylabel("number of nulls", fontsize=MED_TEXT)
    ax.set_xlabel("field", fontsize=MED_TEXT)
    ax.set_title(title, fontsize=BIG_TEXT)
    return ax