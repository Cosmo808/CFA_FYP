from sklearn.mixture import BayesianGaussianMixture
import joblib
import os


class BGM:
    def __init__(self, n_components, prior, max_iter, tol):
        self.n_components = n_components
        self.init_params = 'kmeans'
        self.weight_concentration_prior_type = 'dirichlet_process'
        self.weight_concentration_prior = prior
        self.max_iter = max_iter
        self.tol = tol
        self.bgm_model = None

    def fit_bgm(self, input_data):
        self.bgm_model = BayesianGaussianMixture(n_components=self.n_components, init_params=self.init_params,
                                                 weight_concentration_prior=self.weight_concentration_prior,
                                                 weight_concentration_prior_type=self.weight_concentration_prior_type,
                                                 max_iter=self.max_iter, tol=self.tol)
        self.bgm_model.fit(input_data)

    def save_bgm(self, bgm_path):
        if not os.path.exists(bgm_path):
            os.makedirs(bgm_path)
        file_name = str(self.n_components) + '_' + str(self.weight_concentration_prior)
        file_name = os.path.join(bgm_path, file_name)
        joblib.dump(self.bgm_model, file_name)

    def load_bgm(self, bgm_path, n_components, weight_concentration_prior):
        file_name = str(n_components) + '_' + str(weight_concentration_prior)
        file_name = os.path.join(bgm_path, file_name)
        self.bgm_model = joblib.load(file_name)
        return self.bgm_model
