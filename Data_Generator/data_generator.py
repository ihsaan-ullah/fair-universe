#================================
# External Imports
#================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm 
import scipy.stats as st
from sklearn.utils import shuffle

#================================
# Internal Imports
#================================
from distributions import Gaussian, Exponential, Poisson
from systematics import Ben_New, Translation, Scaling
from logger import Logger
from checker import Checker
from constants import (
    DISTRIBUTION_GAUSSIAN, 
    DISTRIBUTION_EXPONENTIAL, 
    DISTRIBUTION_POISSON,
    SYSTEMATIC_BEN_NEW,
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
    
    def __init__(self, settings_dict=None, logs=False):

        #-----------------------------------------------
        # Initialize data members
        #-----------------------------------------------
        self.settings = None
        self.params_distributions = {} 
        self.params_systematics = None 


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

            
            


        self.problem_dimension = self.settings["problem_dimension"]
        self.total_number_of_events = self.settings["total_number_of_events"]

        self.ps = self.settings["probability_class_signal"]
        self.pb = self.settings["probability_class_background"]

        self.number_of_signal_events = int(self.total_number_of_events*self.ps)
        self.number_of_background_events = int(self.total_number_of_events*self.pb)

        if self.load_from_file:
            self.logger.success("Settings JSON File Loaded!")
        else: 
            self.logger.success("Settings Loaded!")

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
        if self.settings["systematics"]["name"] == SYSTEMATIC_BEN_NEW:
            self.params_systematics = Ben_New(self.settings["systematics"])
        elif self.settings["systematics"]["name"] == SYSTEMATIC_TRANSLATION:
            self.params_systematics = Translation(self.settings["systematics"])
        elif self.settings["systematics"]["name"] == SYSTEMATIC_SCALING:
            self.params_systematics = Scaling(self.settings["systematics"])
        else:
            self.logger.error("Invalid Systematics in {}".format(JSON_FILE))
            exit()

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
        signal_data = self.params_distributions["signal"].generate_points(self.number_of_signal_events, self.problem_dimension)
        
        # get background datapoints
        background_data = self.params_distributions["background"].generate_points(self.number_of_background_events, self.problem_dimension)

        self.logger.success("Data Generated!")


        #-----------------------------------------------
        # Apply Systematics
        #-----------------------------------------------

        # signal points
        biased_signal_data = self.params_systematics.apply_systematics(self.problem_dimension, signal_data)


      

        # background points
        biased_background_data = self.params_systematics.apply_systematics(self.problem_dimension, background_data)

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

    def load_settings_and_generate_data(self):
        self.load_settings()
        self.load_distributions()
        self.load_systematics()
        self.generate_data()

    def get_data(self):

        #-----------------------------------------------
        # Check Data Generated
        #-----------------------------------------------
        if self.checker.data_is_not_generated(self.generated_data):
            self.logger.error("Data is not generated. First call `generate_data` function!")
            exit()


        original_set = {"data": self.generated_data, "labels":self.generated_labels}
        biased_set = {"data": self.biased_data, "labels":self.biased_labels}

        return original_set, biased_set
    
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
        if self.checker.data_is_not_generated(self.generated_data):
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
        plt.scatter(signal_points['x1'], signal_points['x2'], color = 'red', alpha=0.5, label="Signal")
        plt.scatter(background_points['x1'], background_points['x2'], color = 'green', alpha=0.5, label="Background")
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

    def visualize_data_with_contour_2d(self):

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

        
        signal_points = self.generated_dataframe[self.generated_dataframe['y'] == SIGNAL_LABEL].drop('y', axis=1)
        background_points = self.generated_dataframe[self.generated_dataframe['y'] == BACKGROUND_LABEL].drop('y', axis=1)
        

        figure = plt.figure(figsize=(10,8))
        
        plt.scatter(signal_points['x1'], signal_points['x2'], color = 'red', alpha=0.7, label="Signal")
        plt.scatter(background_points['x1'], background_points['x2'], color = 'green', alpha=0.7, label="Background")
     

        # Signal contours
        xx,yy,f,levels = self._get_contours(
            x=signal_points["x1"], 
            y=signal_points["x2"],
            xmin= np.min(signal_points["x1"]),
            xmax= np.max(signal_points["x1"]),
            ymin= np.min(signal_points["x2"]),
            ymax= np.max(signal_points["x2"])
            
        )
        plt.contourf(xx, yy, f, levels, cmap=cm.Reds, alpha=0.4)

        # backgrund contours
        xx,yy,f,levels = self._get_contours(
            x=background_points["x1"], 
            y=background_points["x2"],
            xmin= np.min(background_points["x1"]),
            xmax= np.max(background_points["x1"]),
            ymin= np.min(background_points["x2"]),
            ymax= np.max(background_points["x2"])
            
        )
        plt.contourf(xx, yy, f, levels, cmap=cm.Greens, alpha=0.4)


        
        plt.xlabel("feature 1")
        plt.ylabel("feature 2")
        plt.title("Signal and Background points")
        plt.legend()
        plt.show()

    def _get_contours(self, x, y, xmin, xmax, ymin, ymax):

        
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        step = 0.02
        m = np.amax(f)
        levels = np.arange(0.0, m, step) + step

        return xx, yy, f, levels

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
            self.logger.warning("Directory {} does note exist. Creating directory!".format(directory))
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