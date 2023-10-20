import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from math import sqrt
from math import log

EPSILON = np.finfo(float).eps


# ------------------------------
# Two layerd Neural Network
# ------------------------------
# class TwoLayerNN(nn.Module):

#     def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
#         super(TwoLayerNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(hidden_size2, output_size)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = torch.sigmoid(out)
#         return out


# ------------------------------
# Custom Dataset
# ------------------------------
# class CustomDataset(Dataset):
#     def __init__(self, data, labels=None):
#         self.data = data
#         self.labels = labels

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = {'data': self.data[idx]}
#         if self.labels is not None:
#             sample['labels'] = self.labels[idx]
#         return sample


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
            test_sets=[],
            test_sets_weights=[],
            test_labels=[],
            systematics=None,
            model_name="BDT",
            
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
        self.test_sets = []
        self.test_sets_weights = []
        self.test_labels = []
        for test_set in test_sets:
            self.test_sets.append({"data": test_set})

        for test_set_weights in test_sets_weights:
            self.test_sets_weights.append(test_set_weights) 

        for test_label in test_labels:
            self.test_labels.append(test_label)


        self.systematics = systematics

        # Intialize class variables
        self.validation_sets = None
        self.theta_candidates = np.arange(0, 1, 0.1)
        self.best_theta = 0.95
        self.scaler = StandardScaler()


        # # Hyper params
        # self.num_epochs = 10
        # self.batch_size = 32

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """

        self._generate_validation_sets()
        self._init_model()
        self._train()
        self._choose_theta()
        self._validate()
        self._compute_validation_result()

    def predict(self):
        """
        Params:
            None

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
        """

        self._test()
        self._compute_test_result()

        return {
            "mu_hats": self.mu_hats,
            "delta_mu_hat": self.delta_mu_hat
        }

    def _init_model(self):
        print("[*] - Intialize BDT")
        self.model = [None for a in range(10)]
        for i in range(10):
        
        self.model[i] = XGBClassifier(tree_method="hist",use_label_encoder=False,eval_metric='logloss')


    def _generate_validation_sets(self):
        print("[*] - Generating Validation sets")

        # Calculate the sum of weights for signal and background in the original dataset
        signal_weights = self.train_set["weights"][self.train_set["labels"] == 1].sum()
        background_weights = self.train_set["weights"][self.train_set["labels"] == 0].sum()

        # Split the data into training and validation sets while preserving the proportion of samples with respect to the target variable
        train_df, valid_df, train_label, valid_label, train_weights, valid_weights = train_test_split(
            self.train_set["data"],
            self.train_set["labels"],
            self.train_set["weights"],
            test_size=0.3,
            stratify=self.train_set["labels"]
        )

        # Calculate the sum of weights for signal and background in the training and validation sets
        train_signal_weights = train_weights[train_label == 1].sum()
        train_background_weights = train_weights[train_label == 0].sum()
        valid_signal_weights = valid_weights[valid_label == 1].sum()
        valid_background_weights = valid_weights[valid_label == 0].sum()

        # Balance the sum of weights for signal and background in the training and validation sets
        train_weights[train_label == 1] *= signal_weights / train_signal_weights
        train_weights[train_label == 0] *= background_weights / train_background_weights
        valid_weights[valid_label == 1] *= signal_weights / valid_signal_weights
        valid_weights[valid_label == 0] *= background_weights / valid_background_weights

        train_df = self.scaler.fit_transform(train_df) 

        self.train_set = {
            "data": train_df,
            "labels": train_label,
            "weights": train_weights,
            "settings": self.train_set["settings"]
        }

        self.validation_sets = []
        # Loop 10 times to generate 10 validation sets
        for i in range(0, 10):
            tes = round(np.random.uniform(0.9, 1.10), 2)
            # apply systematics
            valid_with_systematics_temp = self.systematics(
                data=valid_df,
                tes=tes
            ).data

            valid_with_systematics = valid_with_systematics_temp.copy()


            self.validation_sets.append({
                "data": valid_with_systematics,
                "labels": valid_label,
                "weights": valid_weights,
                "settings": self.train_set["settings"],
                "tes" : tes
            })
            del valid_with_systematics_temp

    def _train(self):
        print("[*] - Train Neural Network")

        self._init_model()

        weights_train = self.train_set["weights"].copy()

            
                
        class_weights_train = (weights_train[self.train_set['labels'] == 0].sum(), weights_train[self.train_set['labels'] == 1].sum())

        for i in range(len(class_weights_train)): # loop on B then S target
            #training dataset: equalize number of background and signal
            weights_train[self.train_set['labels'] == i] *= max(class_weights_train)/ class_weights_train[i] 
            #test dataset : increase test weight to compensate for sampling


        print("[*] --- Training Model")
        self._fit(self.train_set['data'], self.train_set['labels'], weights_train)

        print("[*] --- Predicting Train set")
        self.train_set['predictions'] = self._predict(self.train_set['data'], self.best_theta)

        self.train_set['score'] = self._return_score(self.train_set['data'])

        auc_train = roc_auc_score(y_true=self.train_set['labels'], y_score = self.train_set['score'],sample_weight=self.train_set['weights'])      
        print(f"[*] --- AUC train : {auc_train}")


        
    def _fit(self, X, y,w):
        print("[*] --- Fitting Model")
        print("sum of signal" , w[y == 1].sum())    
        print("sum of background" , w[y == 0].sum())
        for i in range(10):
            self.model[i].fit(X, y,sample_weight = w) 
    

    def _return_score(self, X):
        y_predict_ = [0 for i in range(np.shape(X)[:,1])]
        for i in range(10):
            y_predict = self.model[i].predict_proba(X)[:,1]


            y_predict = self.model[i].predict_proba(X)[:,1]
            y_predict_ = y_predict + y_predict_
        
        Y_predict = y_predict_/10
        
        dev  = (Y_predict - y_predict)**2
        
        print(np.sum(dev))
        
        return Y_predict
    
    def _predict(self, X, theta):

        Y_predict = self._return_score(X)
        
        predictions = np.where(Y_predict > theta, 1, 0) 

        return predictions




    def amsasimov_x(self, s, b):
        '''
        This function calculates the Asimov crossection significance for a given number of signal and background events.
        Parameters: s (float) - number of signal events

        Returns:    float - Asimov crossection significance
        '''

        if b<=0 or s<=0:
            return 0
        try:
            return s/sqrt(s+b)
        except ValueError:
            print(1+float(s)/b)
            print (2*((s+b)*log(1+float(s)/b)-s))
        #return s/sqrt(s+b)

    def del_mu_stat(self, s, b):
        '''
        This function calculates the statistical uncertainty on the signal strength.
        Parameters: s (float) - number of signal events
                    b (float) - number of background events
        
        Returns:    float - statistical uncertainty on the signal strength 

        '''
        return (np.sqrt(s + b)/s)

    def get_meta_validation_set(self):

        meta_validation_data = []
        meta_validation_labels = []
        meta_validation_weights = []

        for valid_set in self.validation_sets:
            meta_validation_data.append(valid_set['data'])
            meta_validation_labels = np.concatenate((meta_validation_labels, valid_set['labels']))
            meta_validation_weights = np.concatenate((meta_validation_weights, valid_set['weights']))

        return {
            'data': pd.concat(meta_validation_data),
            'labels': meta_validation_labels,
            'weights': meta_validation_weights
        }

    def _choose_theta(self):

        print("[*] Choose best theta")

        meta_validation_set = self.get_meta_validation_set()
        theta_sigma_squared = []

        # Loop over theta candidates
        # try each theta on meta-validation set
        # choose best theta
        for theta in self.theta_candidates:
            meta_validation_set_df = self.scaler.transform(meta_validation_set["data"])    
            # Get predictions from trained model
            Y_hat_valid = self._predict(meta_validation_set_df, theta)
            Y_valid = meta_validation_set["labels"]

            weights_valid = meta_validation_set["weights"].copy()
            
            # get region of interest
            nu_roi = weights_valid[Y_hat_valid == 1].sum()/10

            weights_valid_signal = weights_valid[Y_valid == 1]  
            weights_valid_bkg = weights_valid[Y_valid == 0]

            Y_hat_valid_signal = Y_hat_valid[Y_valid == 1]  
            Y_hat_valid_bkg = Y_hat_valid[Y_valid == 0] 

            # compute gamma_roi
            gamma_roi = weights_valid_signal[Y_hat_valid_signal == 1].sum()/10


            # compute beta_roi
            beta_roi = weights_valid_bkg[Y_hat_valid_bkg == 1].sum()/10


            # Compute sigma squared mu hat
            sigma_squared_mu_hat = nu_roi/np.square(gamma_roi)

            # get N_ROI from predictions
            theta_sigma_squared.append(sigma_squared_mu_hat)

            print(f"\n[*] --- nu_roi: {nu_roi} --- beta_roi: {beta_roi} --- gamma_roi: {gamma_roi} --- sigma squared: {sigma_squared_mu_hat}")


        # Choose theta with min sigma squared
        try:
            index_of_least_sigma_squared = np.nanargmin(theta_sigma_squared)
        except:
            print("[!] - WARNING! All sigma squared are nan")
            index_of_least_sigma_squared = np.argmin(theta_sigma_squared)

        # self.best_theta = self.theta_candidates[index_of_least_sigma_squared]
        print(f"[*] --- Best theta : {self.best_theta}")

        # self.best_theta = 0.92

        # print(f"[*] --- Best theta forced : {self.best_theta}")


    def _validate(self):
        for valid_set in self.validation_sets:
            valid_set['data'] = self.scaler.transform(valid_set['data'])
            valid_set['predictions'] = self._predict(valid_set['data'], self.best_theta)
            valid_set['score'] = self.model.predict_proba(valid_set['data'])[:,1]


    def _compute_validation_result(self):

        print("[*] - Computing Validation result")

        delta_mu_hats = []
        for valid_set in self.validation_sets:

            Y_hat_train = self.train_set["predictions"]
            Y_train = self.train_set["labels"]
            Y_hat_valid = valid_set["predictions"]
            Y_valid = valid_set["labels"]
            Score_train = self.train_set["score"]
            Score_valid = valid_set["score"]


            auc_valid = roc_auc_score(y_true=valid_set["labels"], y_score=Score_valid,sample_weight=valid_set['weights'])
            print(f"\n[*] --- AUC validation : {auc_valid} --- tes : {valid_set['tes']}")

            # print(f"[*] --- PRI_had_pt : {valid_set['had_pt']}")
            # del Score_valid

            


            weights_train = self.train_set["weights"].copy()
            weights_valid = valid_set["weights"].copy()

            signal_valid = weights_valid[Y_valid == 1]
            background_valid = weights_valid[Y_valid == 0]

            Y_hat_valid_signal = Y_hat_valid[Y_valid == 1]
            Y_hat_valid_bkg = Y_hat_valid[Y_valid == 0]

            signal = signal_valid[Y_hat_valid_signal == 1].sum()
            background = background_valid[Y_hat_valid_bkg == 1].sum()

            significance = self.amsasimov_x(signal,background)  
            print(f"[*] --- Significance : {significance}")

            delta_mu_stat = self.del_mu_stat(signal,background) 
            print(f"[*] --- delta_mu_stat : {delta_mu_stat}")


            # get n_roi
            n_roi = weights_valid[Y_hat_valid == 1].sum()

            # get region of interest
            nu_roi = weights_train[Y_hat_train == 1].sum()

            # compute gamma_roi
            weights_train_signal = weights_train[Y_train == 1]
            weights_train_bkg = weights_train[Y_train == 0]

            Y_hat_train_signal = Y_hat_train[Y_train == 1]
            Y_hat_train_bkg = Y_hat_train[Y_train == 0]

            gamma_roi = weights_train_signal[Y_hat_train_signal == 1].sum()

            # compute beta_roi
            beta_roi = weights_train_bkg[Y_hat_train_bkg == 1].sum()
            if gamma_roi == 0:
                gamma_roi = EPSILON

            # Compute mu_hat
            mu_hat = (n_roi - beta_roi)/gamma_roi

            # Compute delta mu hat (absolute value)
            delta_mu_hat = np.abs(valid_set["settings"]["ground_truth_mu"] - mu_hat)

            delta_mu_hats.append(delta_mu_hat)

            print(f"[*] --- nu_roi: {nu_roi} --- n_roi: {n_roi} --- beta_roi: {beta_roi} --- gamma_roi: {gamma_roi}")

            print(f"[*] --- mu: {np.round(valid_set['settings']['ground_truth_mu'], 4)} --- mu_hat: {np.round(mu_hat, 4)} --- delta_mu_hat: {np.round(delta_mu_hat, 4)}")

        # Average delta mu_hat
        self.delta_mu_hat = np.mean(delta_mu_hats)
        print(f"[*] --- delta_mu_hat (avg): {np.round(self.delta_mu_hat, 4)}")

    def _test(self):
        print("[*] - Testing")
        # Get predictions from trained model
        for test_set in self.test_sets:
            test_df = test_set['data']
            test_df = self.scaler.transform(test_df)
            test_set['predictions'] = self._predict(test_df, self.best_theta)
            test_set['score'] = self.model.predict_proba(test_df)[:,1]

        for test_set, test_set_weights in zip(self.test_sets, self.test_sets_weights):
            test_set['weights'] = test_set_weights
        for test_set, test_label in zip(self.test_sets, self.test_labels):
            test_set['labels'] = test_label





    def _compute_test_result(self):

        print("[*] - Computing Test result")

        mu_hats = []
        for test_set in self.test_sets:

            Y_hat_train = self.train_set["predictions"]
            Y_train = self.train_set["labels"]
            Y_hat_test = test_set["predictions"]
            Y_test = test_set["labels"]
            Score_train = self.train_set["score"]
            Score_test = test_set["score"]

            AUC_test = roc_auc_score(y_true=Y_test, y_score=Score_test,sample_weight=test_set['weights'])


            print(f"\n[*] --- AUC test : {AUC_test}")

            weights_train = self.train_set["weights"].copy()
            weights_test = test_set["weights"].copy()
            
            print(f"[*] --- total weight test: {weights_test.sum()}") 
            print(f"[*] --- total weight train: {weights_train.sum()}")

            Y_hat_test_signal = Y_hat_test[Y_test == 1]
            Y_hat_test_bkg = Y_hat_test[Y_test == 0]    

            weights_test_signal = weights_test[Y_test == 1]
            weights_test_bkg = weights_test[Y_test == 0]    
            
            print(f"[*] --- total test signal : {weights_test_signal.sum()}") 
            print(f"[*] --- total test background train: {weights_test_bkg.sum()}")


            signal = weights_test_signal[Y_hat_test_signal == 1].sum()
            background = weights_test_bkg[Y_hat_test_bkg == 1].sum()

#             signal = weights_test[Y_hat_test == 1].sum()
#             background = weights_test[Y_hat_test == 0].sum()
            
            significance = self.amsasimov_x(signal,background)
            print(f"[*] --- Significance : {significance}")

            delta_mu_stat = self.del_mu_stat(signal,background)

            print(f"[*] --- delta_mu_stat : {delta_mu_stat}")

            # get n_roi

            n_roi = weights_test[Y_hat_test == 1].sum()


            weights_train_signal = weights_train[Y_train == 1]
            weights_train_bkg = weights_train[Y_train == 0]

            Y_hat_train_signal = Y_hat_train[Y_train == 1]
            Y_hat_train_bkg = Y_hat_train[Y_train == 0]

            gamma_roi = weights_train_signal[Y_hat_train_signal == 1].sum()

            # compute beta_roi
            beta_roi = weights_train_bkg[Y_hat_train_bkg == 1].sum()
            if gamma_roi == 0:
                gamma_roi = EPSILON
            
            nu_roi = gamma_roi + beta_roi
            
            # Compute mu_hat
            mu_hat = (n_roi - beta_roi)/gamma_roi

            mu_hats.append(mu_hat)
            print(f"[*] --- mu_hat: {np.round(mu_hat, 4)}")
            
            
            
            print(f"[*] --- signal: {signal} --- background: {background} --- Nu_roi {nu_roi}") 
            print(f"[*] --- mu alter :{(nu_roi - background)/signal}")

        print("\n")

        # Save mu_hat from test
        self.mu_hats = mu_hats