from pd_data_preprocess import Pandas_data
from stat_utils import Stat_utils
import matplotlib.pyplot as plt
from statsmodels.api import load
import statsmodels.api as sm
import numpy as np
import pandas as pd


def plot_points_global_trajectory():
    plt.figure()
    for age_0, age_1, gmv_0, gmv_1 in zip(pd_imputed_age_0, pd_imputed_age_1, reg_gmv_0, reg_gmv_1):
        if np.random.rand() < 0.08:
            plt.plot([age_0, age_1], [gmv_0, gmv_1], alpha=np.random.rand()*0.7)

    # global trajectory
    me_model = load('model/gmv&age_lme_model/delta_age+age_0')
    params = me_model.params
    a = params['delta_age']
    b = params['baseline_age']
    c = params['Intercept']
    x = np.linspace(np.min(pd_imputed_age_0), np.max(pd_imputed_age_1), 200)
    delta_age = np.average(pd_imputed_age_1 - pd_imputed_age_0)
    y = a * delta_age + b * x + c
    # plt.plot(x, y, 'saddlebrown', linewidth=5, alpha=1, label='linear global')

    me_model = load('model/gmv&age_lme_model/delta_age_2+age_0_2+delta_age+age_0')
    params = me_model.params
    a = params['delta_age_2']
    b = params['baseline_age_2']
    c = params['delta_age']
    d = params['baseline_age']
    e = params['Intercept']
    y = a * delta_age ** 2 + b * x ** 2 + c * delta_age + d * x + e
    plt.plot(x, y, 'darkred', linewidth=5, alpha=0.8, label='non-linear global')

    me_model = load('model/gmv&age_lme_model/delta_age_2+age_0_2+delta_age')
    params = me_model.params
    a = params['delta_age_2']
    b = params['baseline_age_2']
    c = params['delta_age']
    d = params['Intercept']
    y = a * delta_age ** 2 + b * x ** 2 + c * delta_age + d
    plt.plot(x, y, 'darkblue', linewidth=5, alpha=0.8, label='non-linear exclude age0')

    # LOWESS
    lowess = sm.nonparametric.lowess
    x = ex_imputed_age
    y = reg_gmv
    frac = 1 / 3
    lowess_flag = False
    if lowess_flag:
        lowess_model = lowess(y, x, frac=frac)
        np.save('model/reg_gmv&age_lowess_np/lowess_{}'.format(str(np.round(frac, 3))), lowess_model)
    lowess_model = np.load('model/reg_gmv&age_lowess_np/lowess_{}.npy'.format(str(np.round(frac, 3))))
    sample_size = int(0.03*len(x))
    lowess_model = pd.DataFrame(lowess_model).drop_duplicates().sample(n=sample_size).sort_values(by=0)
    # plt.plot(lowess_model[0], lowess_model[1], 'midnightblue', linewidth=5, alpha=0.9, label='LOWESS')

    plt.title('GMV progression across age', fontsize=15)
    plt.xlabel('Age / year', fontsize=15)
    plt.ylabel('GMV regressing out sex and ethnicity', fontsize=15)
    plt.xlim([45, 85])
    plt.ylim([6.5e+5, 9.5e+5])
    plt.legend()


def plot_random_coefficient():
    plt.figure()
    corr = params['Group x delta_age Cov'] / (params['Group Var'] * params['delta_age Var']) ** 0.5
    random_effects = list(me_model.random_effects.items())
    random_intercept = []
    random_slope_delta_age = []

    for result in random_effects:
        x = result[1]['Group']
        y = result[1]['delta_age']
        random_intercept.append(x)
        random_slope_delta_age.append(y)
        if np.random.rand() < 0.05:
            plt.scatter(x, y, alpha=np.random.rand())
    plt.title('Correlation between random coefficients', fontsize=15)
    plt.xlabel('random intercept (baseline status)', fontsize=15)
    plt.ylabel('random slope (rate of change per year)', fontsize=15)
    plt.xlim([-170000, 170000])
    plt.ylim([-5000, 5000])
    plt.annotate('Pearson Correlation = {}'.format(np.round(corr, 3)), xy=(1, 0), xycoords='axes fraction',
                 horizontalalignment='right', verticalalignment='bottom', fontsize=15)

    stat.linear_regression_params(np.array(random_intercept).reshape(-1, 1), random_slope_delta_age)
    pred_slope = stat.lr_prediction
    plt.plot(random_intercept, pred_slope, 'purple', linewidth=6, alpha=0.7)
    plt.plot([-170000, 170000], [0, 0], '--', color='black', alpha=0.5)
    plt.plot([0, 0], [-5000, 5000], '--', color='black', alpha=0.5)


if __name__ == "__main__":
    data = Pandas_data()
    stat = Stat_utils()
    pd_age = data.age.iloc[:, 2:4]
    pd_gmv = data.gmv
    pd_sex = data.sex
    pd_eth = data.eth

    pd_age_gmv_sex_eth = pd.concat([pd_age, pd_gmv, pd_sex, pd_eth], axis=1)    # 502411
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['gmv_2'])    # 42806
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['sex'])    # 41766
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['eth_0'])    # 41758
    pd_index = pd_age_gmv_sex_eth.index

    imputed_data = np.load('data/age_gmv_imputation/cov_imputation.npy')
    pd_imputed_data = pd.DataFrame(data=imputed_data, index=pd_index,
                                   columns=['age_2', 'age_3', 'gmv_2', 'gmv_3', 'sex', 'eth_0'])

    # expand age: age_ij
    pd_imputed_age_0 = pd_imputed_data['age_2']
    pd_imputed_age_1 = pd_imputed_data['age_3']
    ex_imputed_age = pd.concat([pd_imputed_age_0, pd_imputed_age_1])

    # expand delta age: age_ij - age_i0
    ex_imputed_baseline_age = pd.concat([pd_imputed_age_0, pd_imputed_age_0])
    ex_imputed_delta_age = ex_imputed_age - ex_imputed_baseline_age

    # expand gmv: gmv_ij
    pd_imputed_gmv_0 = pd_imputed_data['gmv_2']
    pd_imputed_gmv_1 = pd_imputed_data['gmv_3']
    ex_imputed_gmv = pd.concat([pd_imputed_gmv_0, pd_imputed_gmv_1])

    # expand baseline gmv: gmv_i0
    ex_imputed_baseline_gmv = pd.concat([pd_imputed_gmv_0, pd_imputed_gmv_0])

    # expand sex
    pd_imputed_sex = pd_imputed_data['sex']
    ex_imputed_sex = pd.concat([pd_imputed_sex, pd_imputed_sex])

    # expand ethnicity
    pd_imputed_eth = pd_imputed_data['eth_0']
    ex_imputed_eth = pd.concat([pd_imputed_eth, pd_imputed_eth])

    # total expanded data
    pd_ex_data = pd.concat([ex_imputed_age, ex_imputed_delta_age, ex_imputed_gmv,
                            ex_imputed_sex, ex_imputed_eth, ex_imputed_baseline_age], axis=1)
    pd_ex_data = pd_ex_data.rename(columns={0: 'age', 1: 'delta_age', 2: 'gmv',
                                            'eth_0': 'ethnicity', 'age_2': 'baseline_age'})

    pd_ex_data = pd.concat([pd_ex_data, pd_ex_data['age'] ** 2], axis=1)
    pd_ex_data = pd_ex_data.set_axis([*pd_ex_data.columns[:-1], 'age_2'], axis=1, inplace=False)
    pd_ex_data = pd.concat([pd_ex_data, pd_ex_data['delta_age'] ** 2], axis=1)
    pd_ex_data = pd_ex_data.set_axis([*pd_ex_data.columns[:-1], 'delta_age_2'], axis=1, inplace=False)

    me_model = load('model/gmv&age_lme_model/delta_age+age_0')
    params = me_model.params

    # regress out covariates
    # reg_gmv = (B_0 + B_1 * delta_age) + b_0
    reg_gmv = pd_ex_data['gmv'] - (params['sex'] * pd_ex_data['sex'] + params['ethnicity'] * pd_ex_data['ethnicity'])
    reg_gmv_0 = reg_gmv.iloc[:int(len(reg_gmv) / 2)]
    reg_gmv_1 = reg_gmv.iloc[int(len(reg_gmv) / 2):]

    plot_points_global_trajectory()
    plt.show()
