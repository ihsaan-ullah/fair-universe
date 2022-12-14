#================================
# External Imports
#================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#================================
# Internal Imports
#================================
from distributions import Gaussian, Exponential, Poisson
from systematics import Ben, Translation, Scaling
from logger import Logger
from checker import Checker
from constants import (
    DISTRIBUTION_GAUSSIAN, 
    DISTRIBUTION_EXPONENTIAL, 
    DISTRIBUTION_POISSON,
    SYSTEMATIC_BEN,
    SYSTEMATIC_TRANSLATION,
    SYSTEMATIC_SCALING,
    SIGNAL_LABEL,
    BACKGROUND_LABEL,
    JSON_FILE,
    CSV_FILE
)


#================================
# Data Generation Class
#================================
class DataGenerator:
    
    def __init__(self):

        #-----------------------------------------------
        # Initialize data members
        #-----------------------------------------------
        self.settings = None
        self.params_distributions = {} 
        self.params_systematics = None 
        self.generated_dataframe = None

        self.problem_dimension = None
        self.number_of_events = None

        
        #-----------------------------------------------
        # Initialize logger class
        #-----------------------------------------------
        self.logger = Logger()

        #-----------------------------------------------
        # Initialize checks class
        #-----------------------------------------------
        self.checker = Checker()

    def load_settings(self):

        #-----------------------------------------------
        # Load JSON settings file
        #-----------------------------------------------
        if not os.path.exists(JSON_FILE):
            self.logger.error("{} file does not exist!".format(JSON_FILE))
            exit() 
        f = open(JSON_FILE)
        self.settings = json.load(f)
        f.close()


        self.problem_dimension = self.settings["problem_dimension"]
        self.number_of_events = self.settings["number_of_events"]

        self.logger.success("Settings JSON File Loaded!")

    def load_distributions(self):

        #-----------------------------------------------
        # Check settings loaded
        #-----------------------------------------------
        if self.checker.settings_is_not_loaded(self.settings):
            self.logger.error("{} is not loaded. First call `load_settings` function!".format(JSON_FILE))
            exit()
        

        #-----------------------------------------------
        # Setting signal distribution
        #-----------------------------------------------
        if self.settings["signal_distribution"]["name"] == DISTRIBUTION_GAUSSIAN:
                signal_distribution = Gaussian(self.settings["signal_distribution"])
        elif self.settings["signal_distribution"]["name"] == DISTRIBUTION_POISSON:
                signal_distribution = Poisson(self.settings["signal_distribution"])
        else:
                self.logger.error("Invalid Signal Distribution in {}".format(JSON_FILE))
                exit()

        #-----------------------------------------------
        # Setting background distribution
        #-----------------------------------------------
        if self.settings["background_distribution"]["name"] == DISTRIBUTION_GAUSSIAN:
                background_distribution = Gaussian(self.settings["background_distribution"])
        elif self.settings["background_distribution"]["name"] == DISTRIBUTION_EXPONENTIAL:
                background_distribution = Exponential(self.settings["background_distribution"])
        else:
                self.logger.error("Invalid Background Distribution in {}".format(JSON_FILE))
                exit()

        self.params_distributions["signal"] = signal_distribution
        self.params_distributions["background"] = background_distribution

        self.logger.success("Distributions Loaded!")
     
    def load_systematics(self):

        #-----------------------------------------------
        # Check settings loaded
        #-----------------------------------------------
        if self.checker.settings_is_not_loaded(self.settings):
            self.logger.error("{} is not loaded. First call `load_settings` function!".format(JSON_FILE))
            exit()

        #-----------------------------------------------
        # Setting systematics
        #-----------------------------------------------
        if self.settings["systematics"]["name"] == SYSTEMATIC_BEN:
            self.params_systematics = Ben(self.settings["systematics"])
        elif self.settings["systematics"]["name"] == SYSTEMATIC_TRANSLATION:
            self.params_systematics = Translation(self.settings["systematics"])
        elif self.settings["systematics"]["name"] == SYSTEMATIC_SCALING:
            self.params_systematics = Scaling(self.settings["systematics"])
        else:
            self.logger.error("Invalid Systematics in {}".format(JSON_FILE))
            exit()

        self.logger.success("Systematics Loaded!")

    def generate_data(self, apply_systematics=True):

        #-----------------------------------------------
        # Check distributions loaded
        #-----------------------------------------------
        if self.checker.distributions_are_not_loaded(self.params_distributions):
            self.logger.error("Distributions are not loaded. First call `load_distributions` function!")
            exit()

        #-----------------------------------------------
        # Check systematics loaded
        #-----------------------------------------------
        if self.checker.systematics_are_not_loaded(self.params_systematics):
            self.logger.error("Systematics are not loaded. First call `load_systematics` function!")
            exit()


        # column names
        columns = ["x{}".format(i+1) for i in range(0, self.settings["problem_dimension"])]
        columns.append("y")


        #-----------------------------------------------
        # Generate Data
        #-----------------------------------------------
        
        # get signal datapoints
        signal_data = self.params_distributions["signal"].generate_points(self.number_of_events, self.problem_dimension)
        
        # get background datapoints
        background_data = self.params_distributions["background"].generate_points(self.number_of_events, self.problem_dimension)

        self.logger.success("Data Generated!")


        #-----------------------------------------------
        # Apply Systematics
        #-----------------------------------------------

        if apply_systematics:

            # signal points
            signal_data = self.params_systematics.apply_systematics(self.problem_dimension, signal_data)
    
            # background points
            background_data = self.params_systematics.apply_systematics(self.problem_dimension, background_data)

            self.logger.success("Systemtics Applied!")

        #-----------------------------------------------
        # Generate labels
        #-----------------------------------------------

        # stack signal labels with data points
        signal_labels = np.repeat(SIGNAL_LABEL, signal_data.shape[0]).reshape((-1,1))
        signal = np.hstack((signal_data, signal_labels))

        # stack background labels with data points
        background_labels = np.repeat(BACKGROUND_LABEL, background_data.shape[0]).reshape((-1,1))
        background = np.hstack((background_data, background_labels))


        #-----------------------------------------------
        # Create DataFrame from Data
        #-----------------------------------------------

        # create signal df
        signal_df = pd.DataFrame(signal, columns = columns)
       
        # create background df
        background_df = pd.DataFrame(background, columns = columns)

        #-----------------------------------------------
        # Combine Signal and Background in a DataFrame
        #-----------------------------------------------
        
        # combine dataframe
        self.generated_dataframe = pd.concat([signal_df, background_df])

        # suffle dataframe
        self.generated_dataframe = self.generated_dataframe.sample(frac=1).reset_index(drop=True)

    def get_data(self):

        #-----------------------------------------------
        # Check Data Generated
        #-----------------------------------------------
        if self.checker.data_is_not_generated(self.generated_dataframe):
            self.logger.error("Data is not generated. First call `generate_data` function!")
            exit()

        return self.generated_dataframe
    
    def show_statistics(self):

        #-----------------------------------------------
        # Check Data Generated
        #-----------------------------------------------
        if self.checker.data_is_not_generated(self.generated_dataframe):
            self.logger.error("Data is not generated. First call `generate_data` function!")
            exit()


        print("#===============================#")
        print("# Data Statistics")
        print("#===============================#")
        print("Signal datapoints :", self.generated_dataframe[self.generated_dataframe.y == SIGNAL_LABEL].shape[0])
        print("Background datapoints :", self.generated_dataframe[self.generated_dataframe.y == BACKGROUND_LABEL].shape[0])
        print("---------------")
        print("Total  datapoints :", self.generated_dataframe.shape[0])
        print("---------------")
        print("Total Classes = ", 2)
        print("Signal label :", SIGNAL_LABEL)
        print("Background label :", BACKGROUND_LABEL)
        print("---------------")

    def show_distribution_parameters(self):

        #-----------------------------------------------
        # Check distributions loaded
        #-----------------------------------------------
        if self.checker.distributions_are_not_loaded(self.params_distributions):
            self.logger.error("Distributions are not loaded. First call `load_distributions` function!")
            exit()


        print("#===============================#")
        print("# Distribution Parameters")
        print("#===============================#")
        print("Signal Distribution :")
        print(self.settings["signal_distribution"])
        print("---------------")
        print("Background Distribution :")
        print(self.settings["background_distribution"])
        print("---------------")

    def show_systematics_parameters(self):
        print("#===============================#")
        print("# Systematics Parameters")
        print("#===============================#")
        print(self.settings["systematics"])
        print("---------------")

    def visualize_data(self):
        #-----------------------------------------------
        # Check Data Generated
        #-----------------------------------------------
        if self.checker.data_is_not_generated(self.generated_dataframe):
            self.logger.error("Data is not generated. First call `generate_data` function!")
            exit()

        #-----------------------------------------------
        # Check Problem Dimension
        #-----------------------------------------------
        if self.problem_dimension != 2:
            self.logger.error("Visualization not implemented for dimension other than 2")
            exit()


        signal_points = self.generated_dataframe[self.generated_dataframe['y'] == SIGNAL_LABEL]
        background_points = self.generated_dataframe[self.generated_dataframe['y'] == BACKGROUND_LABEL]
        
        figure = plt.figure(figsize=(8,5))
        plt.scatter(signal_points['x1'], signal_points['x2'], color = 'green', alpha=0.5, label="Signal")
        plt.scatter(background_points['x1'], background_points['x2'], color = 'red', alpha=0.5, label="Background")
        plt.xlabel("feature 1")
        plt.ylabel("feature 2")
        plt.title("Signal and Background points")
        plt.legend()
        plt.show()

    def visualize_distributions_1d(self):

        #-----------------------------------------------
        # Check Data Generated
        #-----------------------------------------------
        if self.checker.data_is_not_generated(self.generated_dataframe):
            self.logger.error("Data is not generated. First call `generate_data` function!")
            exit()

        #-----------------------------------------------
        # Check Problem Dimension
        #-----------------------------------------------
        if self.problem_dimension != 2:
            self.logger.error("Visualization not implemented for dimension other than 2")
            exit()

        
        signal_points = self.generated_dataframe[self.generated_dataframe['y'] == SIGNAL_LABEL]
        background_points = self.generated_dataframe[self.generated_dataframe['y'] == BACKGROUND_LABEL]
        
        fig, axs = plt.subplots(2, 2, figsize =(12, 10), tight_layout = True)

        axs[0][0].hist(signal_points['x1'], bins=10)
        axs[0][0].set_xlabel("feature 1")
        axs[0][0].set_ylabel("density")
        axs[0][0].set_title("Signal Density Feature 1")
        axs[0][1].hist(signal_points['x2'], bins=10)
        axs[0][1].set_xlabel("feature 2")
        axs[0][1].set_ylabel("density")
        axs[0][1].set_title("Signal Density Feature 2")
        
        axs[1][0].hist(background_points['x1'], bins=10)
        axs[1][0].set_xlabel("feature 1")
        axs[1][0].set_ylabel("density")
        axs[1][0].set_title("Background Density Feature 1")
        axs[1][1].hist(background_points['x2'], bins=10)
        axs[1][1].set_xlabel("feature 2")
        axs[1][1].set_ylabel("density")
        axs[1][1].set_title("Background Density Feature 1")

        plt.show()

    def visualize_distributions_2d(self):

        #-----------------------------------------------
        # Check Data Generated
        #-----------------------------------------------
        if self.checker.data_is_not_generated(self.generated_dataframe):
            self.logger.error("Data is not generated. First call `generate_data` function!")
            exit()

        #-----------------------------------------------
        # Check Problem Dimension
        #-----------------------------------------------
        if self.problem_dimension != 2:
            self.logger.error("Visualization not implemented for dimension other than 2")
            exit()

        
        signal_points = self.generated_dataframe[self.generated_dataframe['y'] == SIGNAL_LABEL]
        background_points = self.generated_dataframe[self.generated_dataframe['y'] == BACKGROUND_LABEL]
        
        figure = plt.figure(figsize=(8,5))
        plt.hist2d(signal_points["x1"], signal_points["x2"])
        plt.title("Signal Histogram 2D")
        plt.show()

        figure = plt.figure(figsize=(8,5))
        plt.hist2d(background_points["x1"], background_points["x2"])
        plt.title("Background Histogram 2D")
        plt.show()

        

    def save_data(self,):

        #-----------------------------------------------
        # Check Data Generated
        #-----------------------------------------------
        if self.checker.data_is_not_generated(self.generated_dataframe):
            self.logger.error("Data is not generated. First call `generate_data` function!")
            exit()

        self.generated_dataframe.to_csv(CSV_FILE, index=False)

        self.logger.success("Data Saved as CSV in {}".format(CSV_FILE))