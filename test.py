import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


def confidence_ellipse(cov, mean, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transform = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transform + ax.transData)
    return ax.add_patch(ellipse)

dependency_kwargs = np.array([
    [-0.8, 0.5],
    [-0.2, 0.5]
])
mu = 2, -3
scale = 6, 5

x, y = get_correlated_dataset(500, dependency_kwargs, mu, scale)

fig, ax = plt.subplots()
cov = np.cov(x,y)


confidence_ellipse(cov, [2, -3], ax, facecolor='pink', alpha=1, zorder=0)
ax.scatter(x, y, s=3)

plt.show()