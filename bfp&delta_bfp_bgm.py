import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from np_data_preprocess import Numpy_data
from va_baye_gaus_mix import BGM
from plot_utils import Plot_utils

if __name__ == "__main__":
    # hyperparameter
    n_components = 5
    prior = 1e+00
    max_iter = 2000
    tol = 5e-03

    fit_flag = False
    data = Numpy_data()
    bgm = BGM(n_components, prior, max_iter, tol)
    plot = Plot_utils()

    pairwise_age, pairwise_bfp, pairwise_bmi, pairwise_gmv = data.pairwise_data()
    pairwise_age, pairwise_bfp = data.match_pairwise_data(pairwise_age, pairwise_bfp)

    delta_age = pairwise_age[:, 1] - pairwise_age[:, 0]
    delta_bfp = pairwise_bfp[:, 1] - pairwise_bfp[:, 0]
    delta_bfp_per_year = delta_bfp / delta_age

    pairwise_data = pairwise_bfp[:, 0:2]
    pairwise_data[:, 1] = delta_bfp_per_year

    bgm_path = 'model/bfp&delta_bfp_bgm_model'
    if fit_flag:
        bgm.fit_bgm(pairwise_data)
        bgm.save_bgm(bgm_path)

    bgm_model = bgm.load_bgm(bgm_path, n_components, prior)

    weights = bgm_model.weights_
    labels = bgm_model.predict(pairwise_data)
    means = bgm_model.means_  # (n_components, n_features)
    covariances = bgm_model.covariances_  # (n_components, n_features, n_features)

    fig, ax = plt.subplots(1, 1)
    colors = cm.cmaps_listed.get('plasma')
    colors = colors(np.linspace(0, 0.8, len(weights)))
    scatter_num = 1000
    for x, y, label in zip(pairwise_data[:scatter_num, 0], pairwise_data[:scatter_num, 1], labels[:scatter_num]):
        plt.scatter(x, y, color=colors[label])

    color_transparency_weight = (np.array(weights) - np.min(weights) + 0.01) / (
                np.max(weights) - np.min(weights) + 0.01)
    pearson_corr = []
    for i in range(len(means)):
        mean = means[i]
        cov = covariances[i]
        pearson = plot.confidence_ellipse(cov, mean, ax, facecolor=colors[i],
                                          alpha=color_transparency_weight[i], label=i)
        pearson_corr.append(pearson)
        ax.legend()

    fig, axs = plt.subplots(1, 2)
    axs[0].bar(range(len(weights)), weights)
    axs[0].set_title('Weights')
    axs[0].set(xlabel='group')

    axs[1].bar(range(len(pearson_corr)), pearson_corr)
    axs[1].set_title('Correlation')
    axs[1].set(xlabel='group')

    plt.show()
