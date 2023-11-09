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
reference_dir = os.path.join(root_dir, "sample_data/test")
# submitted/predicted lables
prediction_dir = os.path.join(root_dir, "sample_result_submission")
# score file to write score into
score_file = os.path.join(output_dir, "scores.json")


class Scoring:
    def __init__(self):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
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
        # loop over sets (1 value of mu, total 10 sets)
        for i in range(0, 10):
            test_settings_per_mu = []
            # loop over test sets, total 100 test sets
            for j in range(0, 100):
                settings_file = os.path.join(
                    reference_dir, f'set_{i}', "settings", "data.json"
                )
                with open(settings_file) as f:
                    test_settings_per_mu.append(json.load(f))
            self.test_settings.append(test_settings_per_mu)

        print("[✔]")

    def load_ingestion_results(self):
        print("[*] Reading predictions")
        self.ingestion_results = []
        # loop over sets (1 value of mu, total 10 sets)
        for i in range(0, 10):
            results_file = os.path.join(prediction_dir, "result_"+str(i)+".json")
            with open(results_file) as f:
                self.ingestion_results.append(json.load(f))

        print("[✔]")

    def compute_scores(self):
        print("[*] Computing scores")

        # loop over ingestion results
        rmses, maes, intervals, coverages, quantiles_scores = [], [], [], [], []
        for i, (ingestion_result, test_settings) in enumerate(zip(self.ingestion_results, self.test_settings)):

            # just get the first test set mu
            mu = test_settings[0]["ground_truth_mu"]
            mu_hats = ingestion_result["mu_hats"]
            delta_mu_hats = ingestion_result["delta_mu_hats"]
            p16s = ingestion_result["p16"]
            p84s = ingestion_result["p84"]

            set_rmses, set_maes = [], []
            for mu_hat, delta_mu_hat in zip(mu_hats, delta_mu_hats):
                set_rmses.append(self.RMSE_score(mu, mu_hat, delta_mu_hat))
                set_maes.append(self.MAE_score(mu, mu_hat, delta_mu_hat))
            set_interval, set_coverage, set_quantiles_score = self.Quantiles_Score(mu, np.array(p16s), np.array(p84s))

            set_mae = np.mean(set_maes)
            set_rmse = np.mean(set_rmses)

            print("------------------")
            print(f"Set {i}")
            print("------------------")
            print(f"MAE (avg): {set_mae}")
            print(f"RMSE (avg): {set_rmse}")
            print(f"Interval: {set_interval}")
            print(f"Coverage: {set_coverage}")
            print(f"Quantiles Score: {set_quantiles_score}")

            # Save set scores in lists
            rmses.append(set_rmse)
            maes.append(set_mae)
            intervals.append(set_interval)
            coverages.append(set_coverage)
            quantiles_scores.append(set_quantiles_score)

        self.scores_dict = {
            "rmse": np.mean(rmses),
            "mae": np.mean(maes),
            "interval": np.mean(intervals),
            "coverage": np.mean(coverages),
            "quantiles_score": np.mean(quantiles_scores)

        }
        print(f"\n\n[*] --- RMSE: {round(np.mean(rmses), 3)}")
        print(f"[*] --- MAE: {round(np.mean(maes), 3)}")
        print(f"[*] --- Interval: {round(np.mean(intervals), 3)}")
        print(f"[*] --- Coverage: {round(np.mean(coverages), 3)}")
        print(f"[*] --- Quantiles score: {round(np.mean(quantiles_scores), 3)}")

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

    def Quantiles_Score(self, mu, p16, p84, eps=1e-10):

        def Interval(p16, p84):
            """Compute the average of the intervals defined by vectors p16 and p84."""
            return np.mean(p84 - p16)

        def Coverage(mu, p16, p84):
            """Compute the fraction of times scalar mu is within intervals defined by vectors p16 and p84."""
            return_coverage = np.mean((mu >= p16) & (mu <= p84))
            return return_coverage

        def f(x, a1=1/2.1626297577854667, a2=1/9.765625, b1=0, b2=0.36, c1=1.36, c2=1):
            """U-shaped function with mn at 0.68 and f(0.68)=1"""
            if x < 0.68:
                return a1 / ((x - b1) * (c1 - x) + eps)
            else:
                return a2 / ((x - b2) * (c2 - x) + eps)

        coverage = Coverage(mu, p16, p84)
        interval = Interval(p16, p84)
        score = (interval + eps) * f(coverage)
        return interval, coverage, score

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
