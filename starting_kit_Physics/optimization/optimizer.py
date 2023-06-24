import numpy as np
from model_optimize_theta import Model


class Optimizer:
    def __init__(self,
                 thetas,
                 train_sets,
                 test_sets
                 ):

        self.thetas = thetas
        self.train_sets = train_sets
        self.test_sets = test_sets
        self.predictions = []

    def train(self):

        # ---------------------------------
        # Load Model
        # ---------------------------------
        print("\t[*] Loading Model")
        self.model = Model()

        # ---------------------------------
        # Load Over datasets to train
        # ---------------------------------
        for train_set in self.train_sets:

            # Train set
            X_Train = train_set['data']
            Y_Train = train_set['labels']

        # ---------------------------------
        # Train Model
        # ---------------------------------
        print("\t[*] Training Model")
        self.model.fit(X_Train, Y_Train)

    def predict(self):
        # ---------------------------------
        # Get Predictions
        # ---------------------------------
        print("\t[*] Get Predictions")
        for test_set, theta in zip(self.test_sets, self.thetas):
            X_Test = test_set['data']
            self.predictions.append(
                self.model.predict(X_Test, theta)
            )

    def compute_score(self):

        print("\t[*] Compute Scores")

        self.results = []

        for theta, predictions, train_set in zip(self.thetas, self.predictions, self.train_sets):

            Y_Train = train_set["labels"]

            # ---------------------------------
            # Estiamte $\nu_{ROI}$
            # ---------------------------------

            nu_roi = len(predictions[predictions == 1])

            # ---------------------------------
            # Estiamte \gamma_{ROI}
            # ---------------------------------
            # get signal class indexes from labels
            indexes = np.argwhere(Y_Train == 1)
            # get signal class predictions
            signal_predictions = predictions[indexes]
            gamma_roi = len(signal_predictions[signal_predictions == 1])

            # ---------------------------------
            # Estiamte \beta_{ROI}
            # ---------------------------------
            beta_roi = nu_roi - gamma_roi

            # ---------------------------------
            # Compute Score \beta_{ROI}
            # ---------------------------------
            score = self._score(nu_roi, gamma_roi)

            result = {
                "theta": theta,
                "score": score,
                "nu_roi": nu_roi,
                "beta_roi": beta_roi,
                "gamma_roi": gamma_roi
            }

            self.results.append(result)

    def get_best_theta(self):
        print("\t[*] Return Best Theta")
        all_sigma_scores = [res["score"] for res in self.results]
        best_theta_result = self.results[np.argmin(all_sigma_scores)]
        return best_theta_result

    def _score(self, nu_roi, gamma_roi):
        """
        $\sigma^{2}_{\hat{\mu}}$ = $\frac{\nu_{ROI}}{\gamma^{2}_{ROI}}$
        """
        sigma_squared_mu_hat = nu_roi/np.square(gamma_roi)
        return sigma_squared_mu_hat
