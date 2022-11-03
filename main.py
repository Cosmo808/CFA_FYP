import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from data_preprocess import Data
from va_baye_gaus_mix import BGM
from plot_utils import Plot_utils


data = Data()
bgm = BGM(n_components=8)
plot = Plot_utils()

pairwise_age, pairwise_bfp, pairwise_bmi = data.pairwise_data()
pairwise_age, pairwise_bfp = data.combine_pairwise_data(pairwise_age, pairwise_bfp)

delta_age = pairwise_age[:, 1] - pairwise_age[:, 0]
delta_bfp = pairwise_bfp[:, 1] - pairwise_bfp[:, 0]
delta_bfp_per_year = delta_bfp / delta_age

pairwise_data = pairwise_age[:, 0:2]
pairwise_data[:, 1] = delta_bfp_per_year

# bgm_model = bgm.fit_bgm(pairwise_data)
# bgm.save_bgm()

bgm_model = bgm.load_bgm(8, 1e+05)

weights = bgm_model.weights_
labels = bgm_model.predict(pairwise_data)
means = bgm_model.means_    # (n_components, n_features)
covariances = bgm_model.covariances_    # (n_components, n_features, n_features)

fig, ax = plt.subplots()
colors = cm.cmaps_listed.get('plasma')
colors = colors(np.linspace(0, 0.8, len(weights)))
for x, y, label in zip(pairwise_data[:500, 0], pairwise_data[:500, 1], labels[:500]):
    plt.scatter(x, y, color=colors[label])

color_transparency_weight = weights
for i in range(len(means)):
    mean = means[i]
    cov = covariances[i]
    plot.confidence_ellipse(cov, mean, ax, facecolor=colors[i], alpha=weights[i])


# plt.bar(range(len(weights)), weights)
# plt.ylabel('weights')

plt.show()
