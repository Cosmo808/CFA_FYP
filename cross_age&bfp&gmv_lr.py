import numpy as np
import pandas as pd
from pd_data_preprocess import Pandas_data
from va_baye_gaus_mix import BGM
from stat_utils import Stat_utils
from plot_utils import gaussian_distribution
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


if __name__ == "__main__":
    # hyperparameter
    n_components = 5
    prior = 1e+05
    max_iter = 3000
    tol = 1e-03
    
    bgm_fit_flag = True
    data = Pandas_data()
    bgm = BGM(n_components, prior, max_iter, tol)
    stat = Stat_utils()

    pd_age = data.age.iloc[:, 2]
    pd_bfp = data.bfp.iloc[:, 2]
    pd_gmv = data.gmv.iloc[:, 0]

    age_bfp_gmv = pd.concat([pd_age, pd_bfp, pd_gmv], axis=1)    # time point 2
    age_bfp_gmv = age_bfp_gmv.dropna()

    np_age = age_bfp_gmv.iloc[:, 0].to_numpy()
    np_bfp = age_bfp_gmv.iloc[:, 1].to_numpy()
    np_gmv = age_bfp_gmv.iloc[:, 2].to_numpy()
    
    input_data = np_bfp.reshape(-1, 1)
    bgm_path = 'model/cross_age&bfp&gmv_bfp_bgm_model'
    if bgm_fit_flag:
        bgm.fit_bgm(input_data)
        bgm.save_bgm(bgm_path)

    bfp_bgm = bgm.load_bgm(bgm_path, n_components, prior)
    weights = bfp_bgm.weights_
    means = bfp_bgm.means_
    means = means.reshape([1, len(means)])[0]
    variances = bfp_bgm.covariances_

    colors = cm.cmaps_listed.get('plasma')
    colors = colors(np.linspace(0, 0.8, len(weights)))
    color_transparency_weight = (np.array(weights) - np.min(weights) + 0.01) / (
            np.max(weights) - np.min(weights) + 0.01) * 0.8
    fig, ax = plt.subplots(1, 1)
    for mean, var, i in zip(means, variances, range(len(means))):
        mu = mean
        sigma = math.sqrt(var)
        gaussian_distribution(ax, mu, sigma, color=colors[i], alpha=color_transparency_weight[i], label=i)
        ax.legend()

    fig, axs = plt.subplots(1, 2)
    axs[0].bar(range(len(weights)), weights)
    axs[0].set_title('Weights')
    axs[0].set(xlabel='group')
    text = np.round(weights, 4)
    for x, y, t in zip(range(len(weights)), weights, text):
        axs[0].text(x-0.4, 1.01*y, t)

    axs[1].bar(range(len(means)), means)
    axs[1].set_title('Means')
    axs[1].set(xlabel='group')
    text = np.round(means, 2)
    for x, y, t in zip(range(len(means)), means, text):
        axs[1].text(x - 0.4, 1.01 * y, t)

    # cross-sectional: gmv ~ age + bfp
    intercept = []
    beta_age = []
    beta_bfp = []
    means_for_sort = []

    # all data
    X = np.ones(shape=[len(np_age), 2])
    X[:, 0] = np_age
    X[:, 1] = np_bfp
    y = np_gmv
    params_all_data = stat.linear_regression_params(X, y)
    all_intercept = params_all_data['Coefficients'][0]
    all_beta_age = params_all_data['Coefficients'][1]
    all_beta_bfp = params_all_data['Coefficients'][2]
    print('######## All data: sample {} ########'.format(len(y)))
    print(params_all_data, '\n')

    # labelled by bfp
    labels = bfp_bgm.predict(input_data)
    group_index = [[] for i in range(len(weights))]
    for i in range(len(weights)):
        np_labels = np.array(labels)
        index = np.nonzero(np_labels == i)
        index = np.array(index).tolist()[0]
        group_index[i] = index

    for group, index in enumerate(group_index):
        if len(index) < 10:
            print('######## Group {}: not enough sample ########\n'.format(group))
            continue
        else:
            print('######## Group {}: weight {}, sample {} ########'.format(group, np.round(weights[group], 4), len(index)))
        labelled_age = np_age[index]
        labelled_bfp = np_bfp[index]
        labelled_gmv = np_gmv[index]
        X = np.ones(shape=[len(labelled_age), 2])
        X[:, 0] = labelled_age
        X[:, 1] = labelled_bfp
        y = labelled_gmv
        params_all_data = stat.linear_regression_params(X, y)
        intercept.append(params_all_data['Coefficients'][0])
        beta_age.append(params_all_data['Coefficients'][1])
        beta_bfp.append(params_all_data['Coefficients'][2])
        means_for_sort.append(means[group])
        print(params_all_data, '\n')

    fig, axs = plt.subplots(3, 1)
    titles = ['Intercept', 'Beta_age', 'Beta_bfp']
    all_params = [all_intercept, all_beta_age, all_beta_bfp]
    params = [intercept, beta_age, beta_bfp]

    # sorted by mean
    sort_index = np.argsort(means_for_sort)
    for i in range(len(params)):
        param = np.array(params[i])
        sorted_param = param[sort_index]
        params[i] = sorted_param

    for ax, all_param, param, title in zip(axs, all_params, params, titles):
        ax.plot(all_param, '*')
        ax.plot(range(1, len(param) + 1), param, '.-')
        ax.set_title(title)
        text = np.round(param, 1)
        # for x, y, t in zip(range(len(param)), param, text):
        #     ax.text(x, y, t)
    plt.show()
