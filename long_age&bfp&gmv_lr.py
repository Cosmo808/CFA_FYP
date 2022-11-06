import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pd_data_preprocess import Pandas_data
from va_baye_gaus_mix import BGM
from stat_utils import Stat_utils
from plot_utils import Plot_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


if __name__ == "__main__":
    # hyperparameter
    n_components = 5
    prior = 1e+00
    max_iter = 3000
    tol = 1e-03

    bgm_fit_flag = False
    data = Pandas_data()
    bgm = BGM(n_components, prior, max_iter, tol)
    stat = Stat_utils()
    plot = Plot_utils()

    pd_age_0 = data.age.iloc[:, 2]
    pd_bfp_0_1 = data.bfp.iloc[:, 2:4]
    pd_gmv_0_1 = data.gmv.iloc[:, 0:2]

    age_bfp_gmv = pd.concat([pd_age_0, pd_bfp_0_1, pd_gmv_0_1], axis=1)
    age_bfp_gmv = age_bfp_gmv.dropna()

    np_age_0 = age_bfp_gmv.iloc[:, 0].to_numpy()
    np_bfp_0 = age_bfp_gmv.iloc[:, 1].to_numpy()
    np_bfp_1 = age_bfp_gmv.iloc[:, 2].to_numpy()
    np_gmv_0 = age_bfp_gmv.iloc[:, 3].to_numpy()
    np_gmv_1 = age_bfp_gmv.iloc[:, 4].to_numpy()

    delta_bfp = np_bfp_1 - np_bfp_0
    input_data = delta_bfp.reshape(-1, 1)
    bgm_path = 'model/long_age&bfp&gmv_delta_bfp_bgm_model'
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
        plot.gaussian_distribution(ax, mu, sigma, color=colors[i], alpha=color_transparency_weight[i], label=i)
        ax.legend()

    fig, axs = plt.subplots(1, 2)
    axs[0].bar(range(len(weights)), weights)
    axs[0].set_title('Weights')
    axs[0].set(xlabel='group')
    text = np.round(weights, 4)
    for x, y, t in zip(range(len(weights)), weights, text):
        axs[0].text(x - 0.4, 1.01 * y, t)

    axs[1].bar(range(len(means)), means)
    axs[1].set_title('Means')
    axs[1].set(xlabel='group')
    text = np.round(means, 2)
    for x, y, t in zip(range(len(means)), means, text):
        axs[1].text(x - 0.4, 1.01 * y, t)

    # longitudinal: gmv ~ B * age_0 + B * bfp_0 + B * time_point * delta_bfp_type
    #                     + b * time_point + b * delta_bfp_type
    # B: fixed slope
    # b: random intercept

    # labelled by bfp
    labels = np.array(bfp_bgm.predict(input_data))
    eid = age_bfp_gmv.index.to_numpy()
    ex_labels = np.array([labels, labels]).reshape([1, 2 * len(labels)]) + 1
    ex_eid = np.array([eid, eid]).reshape([2 * len(eid)])
    ex_age_0 = np.array([np_age_0, np_age_0]).reshape([1, 2 * len(np_age_0)])
    ex_bfp_0 = np.array([np_bfp_0, np_bfp_0]).reshape([1, 2 * len(np_bfp_0)])
    time_point = np.array([np.zeros_like(eid), np.ones_like(eid)]).reshape([1, 2 * len(eid)]) + 1
    ex_gmv = np.array([np_gmv_0, np_gmv_1]).reshape([1, 2 * len(np_gmv_0)])
    time_x_type = np.multiply(time_point, ex_labels)

    integrated_data = np.zeros(shape=[len(ex_eid), 7])
    integrated_data[:, 0] = ex_eid
    integrated_data[:, 1] = ex_age_0
    integrated_data[:, 2] = ex_bfp_0
    integrated_data[:, 3] = time_point
    integrated_data[:, 4] = ex_gmv
    integrated_data[:, 5] = ex_labels
    integrated_data[:, 6] = time_x_type
    pd_data = pd.DataFrame(integrated_data,
                           columns=['eid', 'age_0', 'bfp_0', 'time_point', 'gmv', 'delta_bfp_type', 'time_x_type']
                           )
    pd_data = pd_data.set_index('eid')

    me_model = smf.mixedlm('gmv ~ age_0 + bfp_0 + time_x_type',
                           re_formula='1', data=pd_data, groups='delta_bfp_type')
    me_model = me_model.fit()
    print(me_model.summary())
    plt.show()
