# ------------------------------
# Imports
# ------------------------------
import os
from sys import path
import numpy as np
from sklearn.preprocessing import StandardScaler
from systematics import postprocess
from tqdm import tqdm


# ------------------------------
# Import local modules
# ------------------------------
module_dir= os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(module_dir)
path.append(root_dir)
from bootstrap import bootstrap
import multiprocessing


# ------------------------------
# Constants
# ------------------------------
EPSILON = np.finfo(float).eps


def _sigma_asimov(mu,weight_train,label_train):

    gamma_roi = weight_train[label_train==1].sum()
    beta_roi = weight_train[label_train==0].sum()

    return mu*gamma_roi + beta_roi



def calculate_comb_llr(args):
    i, arg_dict = args
    mu = arg_dict["mu"]
    width = arg_dict["width"]
    weight_train = arg_dict["weight_train"]
    label_train = arg_dict["label_train"]
    weight_test= arg_dict["weight_test"]
    train_df = arg_dict["train_df"]
    test_df = arg_dict["test_df"]
    use_CR = arg_dict["use_CR"]

    weight_CR_train = weight_train[(train_df['DER_deltar_lep_had']>=i*width) & (train_df['DER_deltar_lep_had']<(i+1)*width)]
    label_CR_train = label_train[(train_df['DER_deltar_lep_had']>=i*width) & (train_df['DER_deltar_lep_had']<(i+1)*width)]
    sum_data_total_CR = (weight_test[(test_df['DER_deltar_lep_had']>=i*width) & (test_df['DER_deltar_lep_had']<(i+1)*width)]).sum()


    weight_SR_train = weight_train[(train_df['DER_deltar_lep_had']>=(i+1)*width) & (train_df['DER_deltar_lep_had']<(i+2)*width)]
    label_SR_train = label_train[(train_df['DER_deltar_lep_had']>=(i+1)*width) & (train_df['DER_deltar_lep_had']<(i+2)*width)]
    sum_data_total_SR = (weight_test[(test_df['DER_deltar_lep_had']>=(i+1)*width) & (test_df['DER_deltar_lep_had']<(i+2)*width)]).sum()

    sigma_asimov_SR_mu = _sigma_asimov(mu,weight_SR_train,label_SR_train)
    sigma_asimov_SR_mu0 = _sigma_asimov(1.0,weight_SR_train,label_SR_train)
    sigma_asimov_CR_mu = _sigma_asimov(mu,weight_CR_train,label_CR_train)
    sigma_asimov_CR_mu0 = _sigma_asimov(1.0,weight_CR_train,label_CR_train)

    # print(f"[*] ---- sigma_asimov_SR_mu: {sigma_asimov_SR_mu}")
    # print(f"[*] ---- sigma_asimov_SR_mu0: {sigma_asimov_SR_mu0}")
    # print(f"[*] ---- sigma_asimov_CR_mu: {sigma_asimov_CR_mu}")
    # print(f"[*] ---- sigma_asimov_CR_mu0: {sigma_asimov_CR_mu0}")

    hist_llr = 0
    hist_llr_SR = (
        -2
        * sum_data_total_SR
        * np.log((sigma_asimov_SR_mu / sigma_asimov_SR_mu0))
    ) + (2 * (sigma_asimov_SR_mu - sigma_asimov_SR_mu0))

    if use_CR:
        hist_llr_CR = (
            -2
            * sum_data_total_CR
            * np.log((sigma_asimov_CR_mu / sigma_asimov_CR_mu0))
        ) + (2 * (sigma_asimov_CR_mu - sigma_asimov_CR_mu0))
    else:
        hist_llr_CR = 0
    hist_llr = hist_llr_SR + hist_llr_CR
    # print(f"[*] ---- hist_llr: {hist_llr}")
    return hist_llr



# ------------------------------
# Baseline Model
# ------------------------------
class Model():
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init : initialize a classifier
    2) fit : can be used to train a classifier
    3) predict: predict mu_hat and delta_mu_hat

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission 
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods
    """

    def __init__(
            self,
            train_set=None,
            systematics=None,
            model_name="one-bin",

    ):
        """
        Model class constructor

        Params:
            train_set:
                labelled train set

            test_sets:
                unlabelled test sets

            systematics:
                systematics class

            model_name:

                name of the model, default: BDT


        Returns:
            None
        """

        # Set class variables from parameters
        self.model_name = model_name
        self.train_set = train_set
        self.systematics = systematics

        # Intialize class variables
        self.force_correction = 0
        self.scaler = StandardScaler()

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """

        self._init_model()
        self._predict()
    def predict(self, test_set):
        """
        Params:
            None

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hats
                - delta_mu_hats
                - p16
                - p84
        """

        self._test(test_set)

        return {
            "mu_hat": self.mu_hat,
            "delta_mu_hat": self.delta_mu_hat,
            "p16": self.p16,
            "p84": self.p84
        }

    def _init_model(self):
        print("[*] - Intialize model")
        train_df = self.train_set["data"].copy()
        train_df['weights'] = self.train_set["weights"].copy()
        train_df['labels'] = self.train_set["labels"].copy()

        train_df = postprocess(train_df)

        self.train_set["labels"] = train_df.pop("labels")
        self.train_set["weights"] = train_df.pop("weights")
        self.train_set["data"] = train_df.copy()

    def _fit(self):
        print("[*] --- Fitting Model")

    def _predict(self):
        print("[*] --- Predicting")
        train_weights = self.train_set["weights"].copy()
        train_labels = self.train_set["labels"].copy()
        # self.gamma_roi = (train_weights*(train_labels)).sum()
        # self.beta_roi = (train_weights*(1-train_labels)).sum()   

        # print(f"[*] --- gamma_roi: {self.gamma_roi}")
        # print(f"[*] --- beta_roi: {self.beta_roi}")

        # significance = self.gamma_roi / np.sqrt(self.gamma_roi + self.beta_roi)
        # print(f"[*] --- significance: {significance}")
        mu_hat, mu_p16, mu_p84 = self._compute_result(train_weights,self.train_set["data"].copy())
        delta_mu_hat = mu_p84 - mu_p16
        val = mu_hat - 1.0
        print(f"[*] --- mu_hat: {mu_hat}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")

        self.force_correction =  val


    def calculate_NLL(self,weights_test,test_df,mu_scan,use_CR=False):
        comb_llr = 0
        num_bins = 10
        full_width = self.train_set["data"]['DER_deltar_lep_had'].max() - self.train_set["data"]['DER_deltar_lep_had'].min()
        width = full_width/(num_bins+2)

        print(f"[*] --- width: {width}")
        print(f"[*] --- full_width: {full_width}")
        print(f"[*] --- num_bins: {num_bins}")

        comb_llr_mu_list = []
        for mu in tqdm(mu_scan):

            arg_dict = {
                "mu": mu,
                "width": width,
                "weight_train": self.train_set["weights"].copy(),
                "label_train": self.train_set["labels"].copy(),
                "weight_test": weights_test.copy(),
                "train_df":self.train_set["data"],
                "test_df": test_df,
                "use_CR": use_CR
            }
            arg_list = [(i, arg_dict) for i in range(num_bins)]
            comb_llr_bin = []

            for i in (range(num_bins)):
                comb_llr_bin.append(calculate_comb_llr(arg_list[i]))
            
            comb_llr_bin = np.array(comb_llr_bin)
            comb_llr_mu = comb_llr_bin.sum()
            comb_llr_mu_list.append(comb_llr_mu)

            print(f"[*] --- comb_llr_mu: {comb_llr_bin}")

        comb_llr = np.array(comb_llr_mu_list)

        # comb_llr = comb_llr/num_bins
        comb_llr = comb_llr - np.amin(comb_llr)

        return comb_llr

    
    def _compute_result(self,weights_test,test_df):
        mu_scan = np.linspace(0, 4, 3)
        nll = self.calculate_NLL(weights_test,test_df,mu_scan,use_CR=True)
        hist_llr = np.array(nll)
        print(f"[*] --- hist_llr: {hist_llr}")
        print(f"[*] --- mu_scan: {mu_scan}")
        if (mu_scan[np.where((hist_llr <= 1.0) & (hist_llr >= 0.0))].size == 0):
            p16 = 0
            p84 = 0
            mu = 0
        else:
            p16 = min(mu_scan[np.where((hist_llr <= 1.0) & (hist_llr >= 0.0))])
            p84 = max(mu_scan[np.where((hist_llr <= 1.0) & (hist_llr >= 0.0))]) 
            mu = mu_scan[np.argmin(hist_llr)]

        mu = mu - self.force_correction
        p16 = p16 - self.force_correction
        p84 = p84 - self.force_correction

        return mu, p16, p84
    
    def _test(self, test_set=None):
        print("[*] - Testing")
        weights_test = test_set["weights"].copy()
        test_df = test_set["data"].copy()

        print(f"[*] --- weights_test: {weights_test.sum()}")

        print(f"[*] --- test_df: {test_df.shape}")
        print(f"[*] --- weights_shape: {weights_test.shape}")

        mu_hat, mu_p16, mu_p84 = self._compute_result(weights_test,test_df)
        delta_mu_hat = mu_p84 - mu_p16
        
        print(f"[*] --- mu_hat: {mu_hat}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")

        self.mu_hat = mu_hat
        self.delta_mu_hat = delta_mu_hat
        self.p16 = mu_p16
        self.p84 = mu_p84
