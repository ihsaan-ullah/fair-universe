import numpy as np
from model import Model


class Optimizer:
    def __init__(self,
                 model_setting,
                 theta,
                 train_set
                 ):

        self.model_setting = model_setting
        self.theta = theta
        self.train_set = train_set

    def optimize(self):

        # Train set
        self.X_Train = self.train_set['data']
        self.Y_Train = self.train_set['labels']


        print("\n--------------------------------------------")
        print("[*] Model : {} --- Theta: {}".format(self.model_setting["model_name"], self.theta))
        print("--------------------------------------------")

        # ---------------------------------
        # Load Model
        # ---------------------------------
        print("\t[*] Loading Model")
        self.model = Model(
            self.model_setting["model_name"],
            self.X_Train,
            self.Y_Train,
            self.X_Train,
            self.model_setting["preprocessing"],
            self.model_setting["preprocessing_method"],
            self.theta
        )

        # ---------------------------------
        # Train Model
        # ---------------------------------
        print("\t[*] Training Model")
        self.model.fit()

    def _score(self, nu_roi, gamma_roi):
        """
        $\sigma^{2}_{\hat{\mu}}$ = $\frac{\nu_{ROI}}{\gamma^{2}_{ROI}}$
        """
        sigma_squared_mu_hat = nu_roi/np.square(gamma_roi)
        return sigma_squared_mu_hat

    def get_result(self):

        # ---------------------------------
        # Get Predictions
        # ---------------------------------
        print("\t[*] Get Predictions")
        predictions = self.model.predict()

        # ---------------------------------
        # Estiamte $\nu_{ROI}$
        # ---------------------------------
        print("\t[*] Estimate nu_ROI")
        nu_roi = len(predictions[predictions == 1])

        # ---------------------------------
        # Estiamte \gamma_{ROI}
        # ---------------------------------
        print("\t[*] Estimate gamma_ROI")
        # get signal class indexes from labels
        indexes = np.argwhere(self.Y_Train == 1)
        # get signal class predictions
        signal_predictions = predictions[indexes]
        gamma_roi = len(signal_predictions[signal_predictions == 1])

        # ---------------------------------
        # Estiamte \beta_{ROI}
        # ---------------------------------
        print("\t[*] Estimate beta_ROI")
        beta_roi = nu_roi - gamma_roi

        # ---------------------------------
        # Compute Score \beta_{ROI}
        # ---------------------------------
        print("\t[*] Compute Score: sigma_squred")
        score = self._score(nu_roi, gamma_roi)

        return {
            "theta": self.theta,
            "score": score,
            "nu_roi": nu_roi,
            "beta_roi": beta_roi,
            "gamma_roi": gamma_roi
        }
