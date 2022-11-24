from statsmodels.api import load
from scipy.stats import chi2
import numpy as np
import pandas as pd


if __name__ == "__main__":
    me_linear = load('model/gmv&age_lme_model/delta_age+age_0')
    me_non_linear = load('model/gmv&age_lme_model/delta_age_2+age_0_2+delta_age')

    ll_linear = me_linear.llf
    ll_non_linear = me_non_linear.llf
    LR_statistic = -2 * (ll_linear - ll_non_linear)
    p_value = chi2.sf(LR_statistic, 2)

    print('P-value =', p_value, '\n')

    n = float(me_linear.summary().tables[0].iloc[1, 1])
    k_linear = len(me_linear.params) + 1
    k_non_linear = len(me_non_linear.params) + 1

    aic_linear = 2 * k_linear - 2 * ll_linear
    aic_non_linear = 2 * k_non_linear - 2 * ll_non_linear

    bic_linear = k_linear * np.log(n) - 2 * ll_linear
    bic_non_linear = k_non_linear * np.log(n) - 2 * ll_non_linear

    print('Non-linear - Linear\nAIC: {}\nBIC: {}'.format(np.round(aic_non_linear - aic_linear, 3),
                                                         np.round(bic_non_linear - bic_linear, 3)))
