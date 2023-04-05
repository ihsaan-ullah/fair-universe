#================================
# Imports
#================================
import numpy as np
from math import cos, sin, radians
from constants import (
    GAUSSIAN_GENERETOR_TYPE_NORMAL,
    GAUSSIAN_GENERETOR_TYPE_MULTIVARIATE
)

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
            sigma = distribution["sigma"],
        )
        self.generator_type = distribution["generator"]
        self.angle_rotation = distribution["angle_rotation"]


    def generate_points(self, number_of_events, problem_dimension):
        """
        This function generates datapoints using Gaussian distribution
        """

        if self.generator_type == GAUSSIAN_GENERETOR_TYPE_NORMAL:

            # initialize vector with required dimension
            points = np.zeros((number_of_events, problem_dimension))
            
            # loop over problem dimension to generate each dimension
            for i in range(0, problem_dimension):
                points[:, i] = np.array(np.random.normal(self.mu[i],self.sigma[i], number_of_events))
                
            return points
        else:

            rotation_matrix = np.array([
                [round(cos(radians(self.angle_rotation)) ,2), round(-sin(radians(self.angle_rotation)) ,2)],
                [round(sin(radians(self.angle_rotation)) ,2), round(cos(radians(self.angle_rotation)) ,2)]
            ])

            covariance_matrix = np.array([
                [self.sigma[0], 0],
                [0, self.sigma[1]]
            ])

            rotated_covariance_matrix = np.matmul(np.matmul(rotation_matrix,covariance_matrix),rotation_matrix.transpose())

            points = np.random.multivariate_normal(mean=self.mu, cov=rotated_covariance_matrix, size=number_of_events)

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
        