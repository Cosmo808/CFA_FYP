import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


class Plot_utils:
    @staticmethod
    def confidence_ellipse(cov, mean, ax, facecolor, alpha, n_std=3.0, **kwargs):
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, edgecolor='purple',
                          zorder=0, alpha=alpha, **kwargs)

        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]

        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]

        transform = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transform + ax.transData)
        ax.add_patch(ellipse)
        return pearson
