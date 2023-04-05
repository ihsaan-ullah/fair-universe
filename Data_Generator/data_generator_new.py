#================================
# External Imports
#================================
import os
import json
import numpy as np
import pandas as pd
from math import cos,sin,radians
import matplotlib.pyplot as plt
from matplotlib import cm 
import scipy.stats as st
from sklearn.utils import shuffle

#================================
# Internal Imports
#================================
from distributions import Gaussian
from systematics import Translation,Scaling
from logger import Logger
from checker import Checker
from constants import (
    DISTRIBUTION_GAUSSIAN, 
    SYSTEMATIC_TRANSLATION,
    SYSTEMATIC_SCALING,
    SIGNAL_LABEL,
    BACKGROUND_LABEL,
    JSON_FILE
)


#================================
# Data Generation Class
#================================
class DataGenerator:
    
    def __init__(self, settings_dict=None, logs=False):

        if logs : 
            print("############################################")
            print("### Data Generation")
            print("############################################")

        #-----------------------------------------------
        # Initialize data members
        #-----------------------------------------------
        self.settings = None
        self.params_distributions = {} 
        self.params_systematics_translation = None 
        self.params_systematics_scaling = None 


        self.generated_data = None
        self.generated_labels = None

        self.biased_data = None
        self.biased_labels = None

        self.problem_dimension = None
        self.ps, self.pb = None, None 
        self.total_number_of_events = None
        self.number_of_background_events = None 
        self.number_of_signal_events = None

        self.settings=settings_dict
        self.load_from_file = settings_dict is None
        
        #-----------------------------------------------
        # Initialize logger class
        #-----------------------------------------------
        self.logger = Logger(show_logs=logs)

        #-----------------------------------------------
        # Initialize checks class
        #-----------------------------------------------
        self.checker = Checker()

    def load_settings(self):

        #-----------------------------------------------
        # Load JSON settings file
        #-----------------------------------------------
        if self.load_from_file:
            if not os.path.exists(JSON_FILE):
                self.logger.error("{} file does not exist!".format(JSON_FILE))
                exit() 
            f = open(JSON_FILE)
            self.settings = json.load(f)
            f.close()

        if self.load_from_file:
            self.logger.success("Settings JSON File Loaded!")
        else: 
            self.logger.success("Settings Loaded!")
            
            

        #-----------------------------------------------
        # Load parameters from Settings
        #-----------------------------------------------

        self.problem_dimension = self.settings["problem_dimension"]
        self.total_number_of_events = self.settings["total_number_of_events"]

        self.pb = self.settings["p_b"]
        self.ps = 1 - self.pb

        self.number_of_signal_events = int(self.total_number_of_events*self.ps)
        self.number_of_background_events = int(self.total_number_of_events*self.pb)


        
        #-----------------------------------------------
        # Load Background distribution
        #-----------------------------------------------

        background_mu = np.array(self.settings["background_mu"])
        background_sigma = np.array(self.settings["background_sigma"])
        background_distribution = Gaussian({
            "name" : DISTRIBUTION_GAUSSIAN,
            "mu" :  background_mu,
            "sigma" : background_sigma,
            "generator" : self.settings["generator"],
            "angle_rotation" : self.settings.get('background_angle_rotation', None)
        })
        self.logger.success("Background Distributions Loaded!")

        #-----------------------------------------------
        # Load Signal distribution
        #-----------------------------------------------
        theta = self.settings["theta"]
        L = self.settings["L"]
        signal_sigma_scale = self.settings["signal_sigma_scale"]
        signal_sigma =  np.multiply(background_sigma, signal_sigma_scale)
        signal_mu = background_mu + np.array([L*cos(radians(theta)), L*sin(radians(theta))])



        signal_distribution = Gaussian({
            "name" : DISTRIBUTION_GAUSSIAN,
            "mu" :  signal_mu,
            "sigma" : signal_sigma,
            "generator" : self.settings["generator"],
            "angle_rotation" : self.settings.get('signal_angle_rotation', None)
        })
        self.logger.success("Signal Distributions Loaded!")


        self.params_distributions["signal"] = signal_distribution
        self.params_distributions["background"] = background_distribution


        
        #-----------------------------------------------
        # Load Systematics
        #-----------------------------------------------

        self.box_center = signal_mu #np.array(self.settings["box_center"])
        self.box_l = self.settings["box_l"]

        scaling_factor = self.settings["scaling_factor"]
        z_magnitude = self.settings["z_magnitude"]
        alpha = self.settings["alpha"]
        z = np.multiply([round(cos(radians(alpha)) ,2), round(sin(radians(alpha)), 2)], z_magnitude)

        self.params_systematics_translation = Translation({
            "name" : SYSTEMATIC_TRANSLATION,
            "allowed_dimension" : -1,
            "translation_vector" : z

        })
        if scaling_factor > 1:
            self.params_systematics_scaling = Scaling({
            "name" : SYSTEMATIC_SCALING,
            "allowed_dimension" : -1,
            "scaling_vector" : [scaling_factor,scaling_factor]

        })

        self.logger.success("Systematics Loaded!")

    def generate_data(self):

        #-----------------------------------------------
        # Check distributions loaded
        #-----------------------------------------------
        if self.checker.distributions_are_not_loaded(self.params_distributions):
            self.logger.error("Distributions are not loaded. First call `load_distributions` function!")
            exit()

        #-----------------------------------------------
        # Check systematics loaded
        #-----------------------------------------------
        if self.checker.systematics_are_not_loaded(self.params_systematics_translation):
            self.logger.error("Systematics are not loaded. First call `load_systematics` function!")
            exit()


        # column names
        columns = ["x{}".format(i+1) for i in range(0, self.settings["problem_dimension"])]
        columns.append("y")


        #-----------------------------------------------
        # Generate Data
        #-----------------------------------------------
        
        # get signal datapoints
        signal_data = self.params_distributions["signal"].generate_points(self.number_of_signal_events, self.problem_dimension)
        
        # get background datapoints
        background_data = self.params_distributions["background"].generate_points(self.number_of_background_events, self.problem_dimension)

        self.logger.success("Data Generated!")


        #-----------------------------------------------
        # Apply Systematics
        #-----------------------------------------------

        ## Translation
        biased_signal_data = self.params_systematics_translation.apply_systematics(self.problem_dimension, signal_data)
        biased_background_data = self.params_systematics_translation.apply_systematics(self.problem_dimension, background_data)
        self.logger.success("Translation Applied!")

        if self.params_systematics_scaling is not None:
            ## Scaling
            biased_signal_data = self.params_systematics_scaling.apply_systematics(self.problem_dimension, biased_signal_data)
            biased_background_data = self.params_systematics_scaling.apply_systematics(self.problem_dimension, biased_background_data)

            self.logger.success("Scaling Applied!")

        
        self.logger.success("Systemtics Applied!")

        #-----------------------------------------------
        # Generate labels
        #-----------------------------------------------

        # stack signal labels with data points
        signal_labels = np.repeat(SIGNAL_LABEL, signal_data.shape[0]).reshape((-1,1))
        signal_original = np.hstack((signal_data, signal_labels))
        signal_biased = np.hstack((biased_signal_data, signal_labels))


        # stack background labels with data points
        background_labels = np.repeat(BACKGROUND_LABEL, background_data.shape[0]).reshape((-1,1))
        background_original = np.hstack((background_data, background_labels))
        background_biased = np.hstack((biased_background_data, background_labels))


        #-----------------------------------------------
        # Create DataFrame from Data
        #-----------------------------------------------

        # create signal df
        signal_df = pd.DataFrame(signal_original, columns = columns)
         # create signal df biased
        signal_df_biased = pd.DataFrame(signal_biased, columns = columns)
       
        # create background df
        background_df = pd.DataFrame(background_original, columns = columns)
        # create background df biased
        background_df_biased = pd.DataFrame(background_biased, columns = columns)

        #-----------------------------------------------
        # Combine Signal and Background in a DataFrame
        #-----------------------------------------------
        
        # combine dataframe
        generated_dataframe = pd.concat([signal_df, background_df])
        biased_dataframe = pd.concat([signal_df_biased, background_df_biased])


        #-----------------------------------------------
        # Apply box
        #-----------------------------------------------
        if self.box_l > 1:
            generated_dataframe, biased_dataframe = self.box_filter(generated_dataframe, biased_dataframe)

        #-----------------------------------------------
        # Separate original and biased data
        #-----------------------------------------------

        # generated data labels
        self.generated_data = generated_dataframe[generated_dataframe.columns[:-1]]
        self.generated_labels = generated_dataframe["y"].to_numpy()

        # biased data labels
        self.biased_data = biased_dataframe[biased_dataframe.columns[:-1]]
        self.biased_labels = biased_dataframe["y"].to_numpy()


        # shuffle data
        self.generated_data = shuffle(self.generated_data, random_state=33)
        self.generated_labels =shuffle(self.generated_labels, random_state=33)

        self.biased_data = shuffle(self.biased_data, random_state=33)
        self.biased_labels =shuffle(self.biased_labels, random_state=33)

    def get_data(self):

        #-----------------------------------------------
        # Check Data Generated
        #-----------------------------------------------
        if self.checker.data_is_not_generated(self.generated_data):
            self.logger.error("Data is not generated. First call `generate_data` function!")
            exit()


        original_set = {"data": self.generated_data, "labels":self.generated_labels}
        biased_set = {"data": self.biased_data, "labels":self.biased_labels}


        return self.settings, original_set, biased_set

    def box_filter(self, df_org, df_bia):
        
        box_x = [self.box_center[0]-self.box_l, self.box_center[0]+self.box_l]
        box_y = [self.box_center[1]-self.box_l, self.box_center[1]+self.box_l]

        mask_org = (df_org['x1'] >= box_x[0]) & (df_org['x1'] <= box_x[1]) & (df_org['x2'] >= box_y[0]) & (df_org['x2'] <= box_y[1])
        mask_bia = (df_bia['x1'] >= box_x[0]) & (df_bia['x1'] <= box_x[1]) & (df_bia['x2'] >= box_y[0]) & (df_bia['x2'] <= box_y[1])

        df_org = df_org.loc[mask_org]
        df_bia = df_bia.loc[mask_bia]

        df_org.reset_index(inplace=True)
        df_bia.reset_index(inplace=True)

        return df_org[["x1", "x2", "y"]], df_bia[["x1", "x2", "y"]]

        



    def save_data(self, directory, file_index=None):

        #-----------------------------------------------
        # Check Data Generated
        #-----------------------------------------------
        if self.checker.data_is_not_generated(self.generated_data):
            self.logger.error("Data is not generated. First call `generate_data` function!")
            exit()

        #-----------------------------------------------
        # Check Directory Exists
        #-----------------------------------------------
        if not os.path.exists(directory):
            self.logger.warning("Directory {} does not exist. Creating directory!".format(directory))
            os.mkdir(directory)
        train_data_dir = os.path.join(directory, "train", "data")
        train_labels_dir = os.path.join(directory, "train", "labels")
        test_data_dir = os.path.join(directory, "test", "data")
        test_labels_dir = os.path.join(directory, "test", "labels")
        settings_dir = os.path.join(directory, "settings")
        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)
        if not os.path.exists(train_labels_dir):
            os.makedirs(train_labels_dir)
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)
        if not os.path.exists(test_labels_dir):
            os.makedirs(test_labels_dir)
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
        
        if file_index is None:
            train_data_name = "train.csv"
            train_labels_name = "train.labels"
            test_data_name = "test.csv"
            test_labels_name = "test.labels"
            settings_file_name = "settings.json"
        else:
            train_data_name = "train_"+str(file_index)+".csv"
            train_labels_name = "train_"+str(file_index)+".labels"
            test_data_name = "test_"+str(file_index)+".csv"
            test_labels_name = "test_"+str(file_index)+".labels"
            settings_file_name = "settings_"+str(file_index)+".json"

        train_data_file = os.path.join(train_data_dir,train_data_name)
        train_labels_file = os.path.join(train_labels_dir, train_labels_name)
        test_data_file = os.path.join(test_data_dir,test_data_name)
        test_labels_file = os.path.join(test_labels_dir,test_labels_name)
        settings_file = os.path.join(settings_dir,settings_file_name)

        self.generated_data.to_csv(train_data_file, index=False)
        self.biased_data.to_csv(test_data_file, index=False)

        with open(train_labels_file, 'w') as filehandle1:
            for ind, lbl in enumerate(self.generated_labels):
                str_label = str(int(lbl))
                if ind < len(self.generated_labels)-1:
                    filehandle1.write(str_label + "\n")
                else:
                    filehandle1.write(str_label)
        filehandle1.close()
        
        with open(test_labels_file, 'w') as filehandle2:
            for ind, lbl in enumerate(self.biased_labels):
                str_label = str(int(lbl))
                if ind < len(self.biased_labels)-1:
                    filehandle2.write(str_label + "\n")
                else:
                    filehandle2.write(str_label)
        filehandle2.close()

        with open(settings_file, 'w') as fp:
            json.dump(self.settings, fp)
        

        

        self.logger.success("Train and Test data saved!")