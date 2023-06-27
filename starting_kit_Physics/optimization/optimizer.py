import numpy as np
from model_optimize_theta import Model


class Optimizer:
    def __init__(
        self,
        thetas,
        train_set
    ):

        self.thetas = thetas
        self.train_set = train_set
        self.predictions = []

    def train(self):

        # ---------------------------------
        # Load Model
        # ---------------------------------
        print("[*] Loading Model")
        self.model = Model()

        # ---------------------------------
        # Train
        # ---------------------------------

        # Train set
        X_Train = self.train_set['data']
        Y_Train = self.train_set['labels']

        # ---------------------------------
        # Train Model
        # ---------------------------------
        print("[*] Training Model")
        self.model.fit(X_Train, Y_Train)

    def predict(self):
        # ---------------------------------
        # Get Predictions
        # ---------------------------------
        print("[*] Get Predictions")
        for theta in self.thetas:
            X_Train = self.train_set['data']
            self.predictions.append(
                self.model.predict(X_Train, theta)
            )

    def compute_score(self):

        print("[*] Compute Scores")

        self.results = []

        for theta, predictions in zip(self.thetas, self.predictions):

            Y_Train = self.train_set["labels"]

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
        print("[*] Return Best Theta")
        all_sigma_scores = [res["score"] for res in self.results]
        best_theta_result = self.results[np.argmin(all_sigma_scores)]
        return best_theta_result

    def _score(self, nu_roi, gamma_roi):
        """
        $\sigma^{2}_{\hat{\mu}}$ = $\frac{\nu_{ROI}}{\gamma^{2}_{ROI}}$
        """
        sigma_squared_mu_hat = nu_roi/np.square(gamma_roi)
        return sigma_squared_mu_hat
