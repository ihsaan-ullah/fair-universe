from math import cos, sin, radians
import numpy as np
from sklearn.naive_bayes import GaussianNB



def _get_params(setting):
    pb = setting["p_b"]
    ps = round(1-pb,2)
    signal_sigma_scale = setting["signal_sigma_scale"]
    L = setting["L"]
    bg_mu = np.array(setting["background_mu"])
    theta = setting["theta"]
    sg_mu = bg_mu + np.array([L * cos(radians(theta)), L * sin(radians(theta))])
    bg_sigma = setting["background_sigma"]
    sg_sigma = np.multiply(bg_sigma, signal_sigma_scale)
    return pb, ps, bg_mu, bg_sigma, sg_mu, sg_sigma

def get_OBC_predictions_score(setting, X):

    pb, ps, bg_mu, bg_sigma, sg_mu, sg_sigma = _get_params(setting)

    model = GaussianNB()

    model.class_prior_ = np.array([pb,ps])
    model.classes_ = np.array([0., 1.])
    model.n_features_in_ = 2
    model.feature_names_in_ = ["x1", "x2"]

    model.theta_ = np.array([bg_mu,sg_mu])
    model.var_ = np.array([bg_sigma,sg_sigma])

    predictions = model.predict(np.array(X))
    scores = model.predict_proba(np.array(X))[:, 1]

    return predictions, scores

