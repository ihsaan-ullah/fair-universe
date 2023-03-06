#================================
# Imports
#================================
import numpy as np


#================================
#  Distribution Class
#================================
class Distribution:

    def __init__(self, name=None, mu=None, sigma=None, lambdaa=None, cut=None):
        
        # init parameters
        self.name = name 
        self.mu = mu 
        self.sigma = sigma 
        self.lambdaa = lambdaa
        self.cut = cut

#================================
# Gaussian Distribution Class
#================================
class Gaussian(Distribution):
    def __init__(self, distribution):

        """
        name: name of the distribution
        mu : 𝜇 parameter of gaussian distribution
        sigma : 𝜎 parameter of gaussian distribution
        """

        super().__init__(
            name = distribution["name"],
            mu =  distribution["mu"],
            sigma = distribution["sigma"]
        )


    def generate_points(self, number_of_events, problem_dimension):
        """
        This function generates datapoints using Gaussian distribution
        """

        # initialize vector with required dimension
        points = np.zeros((number_of_events, problem_dimension))
        
        # loop over problem dimension to generate each dimension
        for i in range(0, problem_dimension):
            points[:, i] = np.array(np.random.normal(self.mu[i],self.sigma[i], number_of_events))
            
        return points

 

#================================
# Poisson Distribution Class
#================================
class Poisson(Distribution):
    
    def __init__(self, distribution):
        super().__init__(
            name = distribution["name"], 
            lambdaa = distribution["lambda"],
            cut = distribution["cut"]
        )
        
    def generate_points(self, number_of_events, problem_dimension):
        """
        This function generates datapoints using Poisson distribution
        """

        # initialize vector with required dimension
        points = np.zeros((number_of_events, problem_dimension))
        
        # loop over problem dimension to generate each dimension
        for i in range(0, problem_dimension):
            if len(self.cut[i]) == 0:
                points[:, i] = np.array(np.random.poisson(self.lambdaa[i], number_of_events))
            else:
                # get min and max limit of the cut
                min_lim, max_lim = self.cut[i]
                points_i = np.array([])
                # loop over points until points are equial to number of events
                while len(points_i) < number_of_events:
                    # generate points
                    points_generated = np.array(np.random.poisson(self.lambdaa[i], number_of_events))
                    # remove points not in limits
                    points_generated = points_generated [ (points_generated >=min_lim) * (points_generated <= max_lim)]
                    # appemd points to previously generated points
                    points_i  = np.append(points_i, points_generated)
                    # remove points if more than number of events
                    if len(points_i) > number_of_events:
                        points_i = points_i[:number_of_events]
                
                points[:, i] = points_i

        return points


  
#================================
# Exponential Distribution Class
#================================
class Exponential(Distribution):

    def __init__(self, distribution):
        super().__init__(
            name = distribution["name"], 
            lambdaa = distribution["lambda"],
            cut = distribution["cut"]
        )


    def generate_points(self, number_of_events, problem_dimension):
        """
        This function generates datapoints using Exponential distribution
        """

        # initialize vector with required dimension
        points = np.zeros((number_of_events, problem_dimension))
        
        # loop over problem dimension to generate each dimension
        for i in range(0, problem_dimension):
            if len(self.cut[i]) == 0:
                points[:, i] = np.array(np.random.exponential(self.lambdaa[i], number_of_events))
            else:
                # get min and max limit of the cut
                min_lim, max_lim = self.cut[i]
                points_i = np.array([])
                # loop over points until points are equial to number of events
                while len(points_i) < number_of_events:
                    # generate points
                    points_generated = np.array(np.random.exponential(self.lambdaa[i], number_of_events))
                    # remove points not in limits
                    points_generated = points_generated [ (points_generated >=min_lim) * (points_generated <= max_lim)]
                    # appemd points to previously generated points
                    points_i  = np.append(points_i, points_generated)
                    # remove points if more than number of events
                    if len(points_i) > number_of_events:
                        points_i = points_i[:number_of_events]
                
                points[:, i] = points_i
        return points
        