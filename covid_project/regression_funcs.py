from covid_project.data_utils import clean_covid_data, clean_policy_data, prep_policy_data
from covid_project.policy_mappings import policy_dict_v1
import pandas as pd
import statsmodels.api as sm

def fit_ols_model_single_policy(data,
                                policy_name,
                                dep_var,
                                use_const=True):

    """Fit an ols model from statsmodels
    
    Parameters
    ----------
    data
    
    policy_name
    
    dep_var
    
    use_const
    
    Returns
    ---------
    dictionary containing the coefficience (params), standard error of the coefficients (std_err),
    r^2 value (r_squared) and the p values (p_values)
    """
    y = data[('info', dep_var)]
    X = data[policy_name]
    
    if use_const:
        X = sm.add_constant(X)

    model = sm.OLS(y, X)
    results = model.fit()

    results_dict = {
        'r_squared': results.rsquared,
        'p_values': results.pvalues.to_dict(),
        'params': results.params.to_dict(),
        'std_err': results.bse.to_dict()
    }
    
    return results_dict