#================================
# External Imports
#================================
import os
import json
import numpy as np
import pandas as pd

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
    JSON_FILE
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
            return 
        f = open(JSON_FILE)
        self.settings = json.load(f)
        f.close()

        self.logger.success("Settings JSON File Loaded!")

    def load_distributions(self):

        #-----------------------------------------------
        # Check settings loaded
        #-----------------------------------------------
        if self.checker.settings_is_not_loaded(self.settings):
            self.logger.error("{} is not loaded. First call `load_settings` function!".format(JSON_FILE))
            return
        

        problem_dimension = self.settings["problem_dimension"]

        #-----------------------------------------------
        # Setting signal distribution
        #-----------------------------------------------
        if self.settings["signal_distribution"]["name"] == DISTRIBUTION_GAUSSIAN:
                signal_distribution = Gaussian(self.settings["signal_distribution"], problem_dimension)
        elif self.settings["signal_distribution"]["name"] == DISTRIBUTION_POISSON:
                signal_distribution = Poisson(self.settings["signal_distribution"], problem_dimension)
        else:
                self.logger.error("Invalid Signal Distribution in {}".format(JSON_FILE))
                return

        #-----------------------------------------------
        # Setting background distribution
        #-----------------------------------------------
        if self.settings["background_distribution"]["name"] == DISTRIBUTION_GAUSSIAN:
                background_distribution = Gaussian(self.settings["background_distribution"], problem_dimension)
        elif self.settings["background_distribution"]["name"] == DISTRIBUTION_EXPONENTIAL:
                background_distribution = Exponential(self.settings["background_distribution"], problem_dimension)
        else:
                self.logger.error("Invalid Background Distribution in {}".format(JSON_FILE))
                return 

        self.params_distributions["signal"] = signal_distribution
        self.params_distributions["background"] = background_distribution

        self.logger.success("Distributions Loaded!")

        
    def load_systematics(self):

        #-----------------------------------------------
        # Check settings loaded
        #-----------------------------------------------
        if self.checker.settings_is_not_loaded(self.settings):
            self.logger.error("{} is not loaded. First call `load_settings` function!".format(JSON_FILE))
            return

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
            return 

        self.logger.success("Systematics Loaded!")

    def generate_data(self):

        #-----------------------------------------------
        # Check distributions loaded
        #-----------------------------------------------
        if self.checker.distributions_are_not_loaded(self.params_distributions):
            self.logger.error("Distributions are not loaded. First call `load_distributions` function!")
            return

        #-----------------------------------------------
        # Check systematics loaded
        #-----------------------------------------------
        if self.checker.systematics_are_not_loaded(self.params_systematics):
            self.logger.error("Systematics are not loaded. First call `load_systematics` function!")
            return


        # column names
        columns = ["x{}".format(i+1) for i in range(0, self.settings["problem_dimension"])]
        columns.append("label")


        #-----------------------------------------------
        # Generate Signal Data
        #-----------------------------------------------

        # get data points
        signal_data = self.params_distributions["signal"].generate_points()

        
        # stack labels with data points
        signal_labels = np.repeat(SIGNAL_LABEL, signal_data.shape[0]).reshape((-1,1))
        signal = np.hstack((signal_data, signal_labels))
   

        # create signal df
        signal_df = pd.DataFrame(signal, columns = columns)
       
       
        #-----------------------------------------------
        # Generate Background Data
        #-----------------------------------------------

        # get data points
        background_data = self.params_distributions["background"].generate_points()

        # stack labels with data points
        # stack labels with data points
        background_labels = np.repeat(SIGNAL_LABEL, background_data.shape[0]).reshape((-1,1))
        background = np.hstack((background_data, background_labels))

        # create background df
        background_df = pd.DataFrame(background, columns = columns)

        #-----------------------------------------------
        # Combine Signal and Background in a DataFrame with classes
        #-----------------------------------------------
        
        self.generated_dataframe = pd.concat([signal_df, background_df])
        

        self.logger.success("Data Generated!")
        

    def get_data(self):

        #-----------------------------------------------
        # Check Data Generated
        #-----------------------------------------------
        if self.checker.data_is_not_generated(self.generated_dataframe):
            self.logger.error("Data is not generated. First call `generate_data` function!")
            return

        return self.generated_dataframe
    
    def show_statistics(self):
        print("#===============================#")
        print("# Data Statistics")
        print("#===============================#")
        # print("Signal datapoints :", signal_train_df.shape[0])
        # print("Background datapoints :", background_train_df.shape[0])
        # print("---------------")
        # print("Total  datapoints :", signal_train_df.shape[0]+background_train_df.shape[0])
        # print("---------------")
        print("Total Classes = ", 2)
        print("Signal label :", SIGNAL_LABEL)
        print("Background label :", BACKGROUND_LABEL)
        print("---------------")














    def get_dataa(
        self, 
        combine_train_distributions=True,
        print_statistics = False
        ):
        """
        This function generates signal and background data and organize it in dataframes
        """

        #-----------------------------------------------
        # Get signal and background points
        #-----------------------------------------------
        signal_train = self.signal_train.get_points(self.signal_train_events)
        signal_test = self.signal_test.get_points(self.signal_test_events)
        background_train = self.background_train.get_points(self.background_train_events)
        background_test = self.background_test.get_points(self.background_test_events)


        #-----------------------------------------------
        # Combie signal train and test with signal class
        #-----------------------------------------------
        S_train = np.stack((signal_train, np.repeat(self.SIGNAL_LABEL, signal_train.shape)),axis=1)
        S_test = np.stack((signal_test, np.repeat(self.SIGNAL_LABEL, signal_test.shape)),axis=1)

        #-----------------------------------------------
        # Combie background train and test with signal class
        #-----------------------------------------------
        B_train = np.stack((background_train, np.repeat(self.BACKGROUND_LABEL, background_train.shape)),axis=1)
        B_test = np.stack((background_test, np.repeat(self.BACKGROUND_LABEL, background_test.shape)),axis=1)
    

        #-----------------------------------------------
        # Create signal train and test dataframes
        #-----------------------------------------------
        signal_train_df = pd.DataFrame(S_train, columns =['x', 'y'])
        signal_test_df = pd.DataFrame(S_test, columns =['x', 'y'])
        
        #-----------------------------------------------
        # Create background train and test dataframes
        #-----------------------------------------------
        background_train_df = pd.DataFrame(B_train, columns =['x', 'y'])
        background_test_df = pd.DataFrame(B_test, columns =['x', 'y'])


        #-----------------------------------------------
        # Print data statistics
        #-----------------------------------------------
        if print_statistics:
            self.get_data_statistics(
                signal_train_df,
                signal_test_df,
                background_train_df,
                background_test_df
            )

            
        
        if combine_train_distributions:

            # combine train in one df and test in another
            df_train = pd.concat([signal_train_df, background_train_df])
            df_test = pd.concat([signal_test_df, background_test_df])

            # shuffle both dfs
            df_train = df_train.sample(frac=1).reset_index(drop=True)
            df_test = df_test.sample(frac=1).reset_index(drop=True)


            return (
                df_train,
                df_test
            )
        else:

            return (
                signal_train_df,
                signal_test_df,
                background_train_df,
                background_test_df
            )





  
