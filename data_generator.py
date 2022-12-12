#================================
# Imports
#================================
import os
import json
import numpy as np
import pandas as pd

from distributions import Gaussian, Exponential, Poisson
from systematics import Ben, Translation, Scaling
from errors import Errors
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
        params_distributions = None 
        params_systematics = None 
        generated_dataframe = None

        
        #-----------------------------------------------
        # Initialize errors class
        #-----------------------------------------------
        self.e = Errors()



    def load_settings(self):

        #-----------------------------------------------
        # Load JSON settings file
        #-----------------------------------------------
        if not os.path.exists(JSON_FILE):
            self.e.error("{} file does not exist!".format(JSON_FILE))
            return 
        f = open(JSON_FILE)
        self.settings = json.load(f)
        f.close()

    def load_distributions(self):

        #-----------------------------------------------
        # Setting signal distribution
        #-----------------------------------------------
        if self.settings["signal_distribution"]["name"] == DISTRIBUTION_GAUSSIAN:
                self.signal_distribution = Gaussian(self.settings["signal_distribution"])
        elif self.settings["signal_distribution"]["name"] == DISTRIBUTION_POISSON:
                self.signal_distribution = Poisson(self.settings["signal_distribution"])
        else:
                self.e.error("Invalid Signal Distribution in {}".format(JSON_FILE))

        #-----------------------------------------------
        # Setting background distribution
        #-----------------------------------------------
        if self.settings["background_distribution"]["name"] == DISTRIBUTION_GAUSSIAN:
                self.signal_distribution = Gaussian(self.settings["background_distribution"])
        elif self.settings["background_distribution"]["name"] == DISTRIBUTION_EXPONENTIAL:
                self.signal_distribution = Exponential(self.settings["background_distribution"])
        else:
                self.e.error("Invalid Background Distribution in {}".format(JSON_FILE))

    def load_systematics(self):
        pass

    def generate_data(self):
        pass

    def get_data(self):
        pass 
    
    def show_statistics(self):
        print("#===============================#")
        print("# Data Statistics")
        print("#===============================#")
        # print("Signal Train datapoints :", signal_train_df.shape[0])
        # print("Signal Test datapoints :", signal_test_df.shape[0])
        # print("Background Train datapoints :", background_train_df.shape[0])
        # print("Background Test datapoints :", background_test_df.shape[0])
        # print("---------------")
        # print("Total Signal datapoints :", signal_train_df.shape[0]+signal_test_df.shape[0])
        # print("Total Background datapoints :", background_train_df.shape[0]+background_test_df.shape[0])
        # print("---------------")
        # print("Total Train datapoints :", signal_train_df.shape[0]+background_train_df.shape[0])
        # print("Total Test datapoints :", signal_test_df.shape[0]+background_test_df.shape[0])
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





  
