# covid-19-data-analysis
A study on how the COVID-19 responses might be affecting new case and death numbers.

There are 2 notebooks associated with this project: 

### 1) Covid-19 data analysis.ipynb

This is the main notebook associated with the project. It contains a brief discussion of data cleaning, plotting cases and deaths over time at the county, state, and national level, overlays with selected policy implementations, and a correlation between policies and changes in new case and death numbers. 

Policies were correlated with two key metrics in case and death numbers: the difference in new cases and deaths on the day of policy implementation and 14 days later, and the change in curvature of the new cases and new deaths plot- which I refer to as the 'acceleration' of new cases and new deaths.

### 2) Covid-19 data cleaning.ipynb

This is the detailed column-by-column data cleaning process that went into preparing the Covid-19 data analysis notebook, including a detailed analysis of invalid datapoints. 


### Key Findings

- starting policies related to closing houses of worship is correlated with correlated with a large decrease in case acceleration
- stopping mask requirements are correlated with a large decrease in cases, but an increase in case and death acceleration

- The fact that these policies are correlated with and increase or decrease in cases does not imply that they caused the decrease. To gain insight into a causal relationship, I plan to introduce a linear regression step that estimates the impact of each policy type on the overall change in cases and deaths. This is a replication of the work done by [Klimek et al.](https://www.nature.com/articles/s41562-020-01009-0?fbclid=IwAR0RLAeAfgUIpGqwlKD8nHvdtbrj_pPvpG54hPWXaATp0vjtw9z47wUfsjs)