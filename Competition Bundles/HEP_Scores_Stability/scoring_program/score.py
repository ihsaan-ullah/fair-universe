# ------------------------------------------
# Imports
# ------------------------------------------
import os
import numpy as np
import json
from datetime import datetime as dt

# ------------------------------------------
# Default Directories
# ------------------------------------------
# root directory
module_dir= os.path.dirname(os.path.realpath(__file__))

root_dir = os.path.dirname(module_dir)
# Directory to output computed score into
output_dir = os.path.join(root_dir, "scoring_output")
# reference data (test labels)
reference_dir = os.path.join(root_dir, "reference_data")
# submitted/predicted lables
prediction_dir = os.path.join(root_dir, "sample_result_submission")
# score file to write score into
score_file = os.path.join(output_dir, "scores.json")

# ------------------------------------------
# Codabench Directories
# ------------------------------------------
# # root directory
# root_dir = "/app"
# # Directory read predictions and solutions from
# input_dir = os.path.join(root_dir, "input")
# # Directory to output computed score into
# output_dir = os.path.join(root_dir, "output")
# # reference data (test labels)
# reference_dir = os.path.join(input_dir, 'ref')  # Ground truth data
# # submitted/predicted labels
# prediction_dir = os.path.join(input_dir, 'res')
# # score file to write score into
# score_file = os.path.join(output_dir, 'scores.json')


class Scoring:
    def __init__(self):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
        # self.test_labels = None
        self.test_settings = None
        self.ingestion_results = None

        self.scores_dict = {}

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def show_duration(self):
        print("\n---------------------------------")
        print(f"[✔] Total duration: {self.get_duration()}")
        print("---------------------------------")

    def load_test_settings(self):
        print("[*] Reading test settings")
        self.test_settings = []
        for i in range(0, 1):
            settings_file = os.path.join(
                reference_dir, f'set_{1}',"settings", "data.json"
            )
            with open(settings_file) as f:
                self.test_settings.append(json.load(f))

        print("[✔]")

    def load_ingestion_results(self):
        print("[*] Reading predictions")

        results_file = os.path.join(prediction_dir, "result.json")
        with open(results_file) as f:
            self.ingestion_results = json.load(f)

        print("[✔]")

    def compute_scores(self):
        print("[*] Computing scores")

        mus = [test_setting["ground_truth_mu"] for test_setting in self.test_settings]
        mu_hats = self.ingestion_results["mu_hats"]
        delta_mu_hats = self.ingestion_results["delta_mu_hats"]
        p16s = self.ingestion_results["p16"]
        p84s = self.ingestion_results["p84"]

        rmses, maes, ben_saschas = [], [], []
        for i , mu in enumerate(mus):
            for mu_hat, delta_mu_hat in zip(mu_hats, delta_mu_hats):
                rmses.append(self.RMSE_score(mu, mu_hat, delta_mu_hat))
                maes.append(self.MAE_score(mu, mu_hat, delta_mu_hat))
            ben_saschas_score, ben_saschas_coverage = self.Ben_Sasha_score(mu,np.array(p16s),np.array(p84s)) 
            ben_saschas.append(ben_saschas_score)
            print(f"[*] --- Coverage for set_{i}: {ben_saschas_coverage}")
            

        self.scores_dict = {
            "rmse": np.mean(rmses),
            "mae": np.mean(maes),
            "ben_sascha": np.mean(ben_saschas)

        }
        print(f"[*] --- rmse: {round(np.mean(rmses), 3)}")
        print(f"[*] --- mae: {round(np.mean(maes), 3)}")
        print(f"[*] --- ben_sascha: {round(np.mean(ben_saschas), 3)}")

        print("[✔]")

    def RMSE_score(self, mu, mu_hat, delta_mu_hat):
        """Compute the sum of MSE and MSE2."""

        def MSE(mu, mu_hat):
            """Compute the mean squared error between scalar mu and vector mu_hat."""
            return np.mean((mu_hat - mu) ** 2)

        def MSE2(mu, mu_hat, delta_mu_hat):
            """Compute the mean squared error between computed delta_mu = mu_hat - mu and delta_mu_hat."""
            adjusted_diffs = (mu_hat - mu)**2 - delta_mu_hat**2
            return np.mean(adjusted_diffs**2)

        return np.sqrt(MSE(mu, mu_hat) + MSE2(mu, mu_hat, delta_mu_hat))

    def MAE_score(self, mu, mu_hat, delta_mu_hat):
        """Compute the sum of MAE and MAE2."""

        def MAE(mu, mu_hat):
            """Compute the mean absolute error between scalar mu and vector mu_hat."""
            return np.mean(np.abs(mu_hat - mu))

        def MAE2(mu, mu_hat, delta_mu_hat):
            """Compute the mean absolute error based on the provided definitions."""
            adjusted_diffs = np.abs(mu_hat - mu) - delta_mu_hat
            return np.mean(np.abs(adjusted_diffs))

        return MAE(mu, mu_hat) + MAE2(mu, mu_hat, delta_mu_hat)

    def Ben_Sasha_score(self, mu, p16, p84, eps=1e-10):

        def Interval(p16, p84):
            """Compute the average of the intervals defined by vectors p16 and p84."""
            return np.mean(p84 - p16)

        def Coverage(mu, p16, p84):
            """Compute the fraction of times scalar mu is within intervals defined by vectors p16 and p84."""
            return_coverage =  np.mean((mu >= p16) & (mu <= p84))
            return return_coverage

        def f(x, a1=1/2.1626297577854667, a2=1/9.765625, b1=0, b2=0.36, c1=1.36, c2=1):
            """U-shaped function with mn at 0.68 and f(0.68)=1"""
            if x < 0.68:
                return a1 / ((x - b1) * (c1 - x) + eps)
            else:
                return a2 / ((x - b2) * (c2 - x) + eps)
        coverage = Coverage(mu, p16, p84)
        return (Interval(p16, p84) + eps) * f(coverage), coverage

    def write_scores(self):
        print("[*] Writing scores")

        with open(score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))

        print("[✔]")
        pass


if __name__ == "__main__":
    print("############################################")
    print("### Scoring Program")
    print("############################################\n")

    # Init scoring
    scoring = Scoring()

    # Start timer
    scoring.start_timer()

    # Load test settings
    scoring.load_test_settings()

    # Load ingestions results
    scoring.load_ingestion_results()

    # Compute Scores
    scoring.compute_scores()

    # Write scores
    scoring.write_scores()

    # Stop timer
    scoring.stop_timer()

    # Show duration
    scoring.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Scoring Program executed successfully!")
    print("----------------------------------------------\n\n")
