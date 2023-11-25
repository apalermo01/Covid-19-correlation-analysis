# Covid-19 Correlation Analysis
The objective of this project is to experiment with regression analysis and feature engineering to study how the COVID-19 responses might have affected new case and death numbers in the first year of the pandemic. 

Case and death data were sourced from the Johns Hopkins University's COVID-19 tracking project. Policy data was sourced from data.gov. 

The core of this project exists in a series of jupyter notebooks, which are organized as follows:

- 00: high-level overview of the methodology used and current results
- 01: Exploratory data analysis on case and death data
- 02: Exploratory data analysis on policy data
- 03: Making some basic plots (This notebook is mostly for posterity - I started this project shortly after I started learning python and these plots were one of the first things I tried to make)
- 04: checking the average change in cases and deaths 14 days after a policy was implemented
- 05: running the first round of linear regression models (single policy)
- 06: analyzing the results of the runs started in 05 

# Objectives for next round of development

- [ ] run the simplest model (sklearn linear regression) on single policy (training set) -> analyze RMSE values and coefficients for 3 different sets of bins
- [ ] use stats models to get p-values from the same kind of model -> compare p values on the same set of bins
- [ ] implement logic to run forward selection on bins based on rmse.
- [ ] run forward selection on bins for each policy -> look at rmse values (training set)
- [ ] Repeat this process with random forests: start with the same set of 3 bins, compare rmse against linear regression, and analyze the parameters (if I try a decision tree first).
- [ ] Re-run linear regression and decision trees while trying multiple policies
- [ ] start looking into bayesian methods
