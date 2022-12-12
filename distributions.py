#================================
# Imports
#================================
import numpy as np
from logger import Logger

#================================
# Gaussian Distribution Class
#================================
class Gaussian:
    def __init__(self, distribution, problem_dimension):

        # name: name of the distribution 
        self.name = distribution["name"]

        # mu : 𝜇 parameter of gaussian distribution
        self.mu = distribution["mu"]

        # sigma : 𝜎 parameter of gaussian distribution
        self.sigma = distribution["sigma"]

        # number_of_events : number of datapoints to be generated
        self.number_of_events = distribution["number_of_events"]

        # problem_dimension: dimension of generated data
        self.problem_dimension = problem_dimension

        # logger object
        self.logger = Logger()

    def generate_points(self):
        """
        This function generates datapoints using Gaussian distribution
        """

        # initialize vector with required dimension
        points = np.zeros((self.number_of_events, self.problem_dimension))
        
        # loop over problem dimension to generate each dimension
        for i in range(0, self.problem_dimension):
            points[:, i] = np.array(np.random.normal(self.mu[i],self.sigma[i], self.number_of_events))
        

        return points

 

#================================
# Poisson Distribution Class
#================================
class Poisson:
    
    def __init__(self, distribution, problem_dimension):

        # name: name of the distribution 
        self.name = distribution["name"]

        # lambdaa : 𝜆 parameter of poisson distribution
        self.lambdaa = distribution["lambda"]

        # number_of_events : number of datapoints to be generated
        self.number_of_events = distribution["number_of_events"]

        # problem_dimension: dimension of generated data
        self.problem_dimension = problem_dimension

        # logger object
        self.logger = Logger()

        
    def generate_points(self):
        """
        This function generates datapoints using Poisson distribution
        """

        # initialize vector with required dimension
        points = np.zeros((self.number_of_events, self.problem_dimension))
        
        # loop over problem dimension to generate each dimension
        for i in range(0, self.problem_dimension):
            points[:, i] = np.array(np.random.poisson(self.lambdaa[i], self.number_of_events))
        

        return points


  
#================================
# Exponential Distribution Class
#================================
class Exponential:
    def __init__(self, distribution, problem_dimension):

        # name: name of the distribution 
        self.name = distribution["name"]

        # lambdaa : 𝜆 parameter of exponential distribution
        self.lambdaa = distribution["lambda"]

        # number_of_events : number of datapoints to be generated
        self.number_of_events = distribution["number_of_events"]

        # problem_dimension: dimension of generated data
        self.problem_dimension = problem_dimension

        # logger object
        self.logger = Logger()

    def generate_points(self):
        """
        This function generates datapoints using Exponential distribution
        """

        # initialize vector with required dimension
        points = np.zeros((self.number_of_events, self.problem_dimension))
        
        # loop over problem dimension to generate each dimension
        for i in range(0, self.problem_dimension):
            points[:, i] = np.array(np.random.exponential(self.lambdaa[i], self.number_of_events))
        
        return points
        