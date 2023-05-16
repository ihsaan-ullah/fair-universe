# ================================
# Imports
# ================================
import numpy as np
from math import cos, sin, radians
from constants import (
    GAUSSIAN_GENERETOR_TYPE_NORMAL,
    GAUSSIAN_GENERETOR_TYPE_MULTIVARIATE,
    DISTRIBUTION_GAUSSIAN,
    DISTRIBUTION_GAMMA
)


# ================================
#  Distribution Class
# ================================
class Distribution:

    def __init__(self, name=None):

        # init parameters
        self.name = name


# ================================
# Gaussian Distribution Class
# ================================
class Gaussian(Distribution):
    def __init__(self, distribution):

        """
        name: name of the distribution
        mu : 𝜇 parameter of gaussian distribution
        sigma : 𝜎 parameter of gaussian distribution
        """

        super().__init__(
            name=distribution["name"]
        )
        self.mu = distribution["mu"]
        self.sigma = distribution["sigma"]
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
                [round(cos(radians(self.angle_rotation)), 2), round(-sin(radians(self.angle_rotation)), 2)],
                [round(sin(radians(self.angle_rotation)), 2), round(cos(radians(self.angle_rotation)), 2)]
            ])

            covariance_matrix = np.array([
                [self.sigma[0], 0],
                [0, self.sigma[1]]
            ])

            rotated_covariance_matrix = np.matmul(np.matmul(rotation_matrix, covariance_matrix), rotation_matrix.transpose())

            points = np.random.multivariate_normal(mean=self.mu, cov=rotated_covariance_matrix, size=number_of_events)

            return points


# ================================
# Poisson Distribution Class
# ================================
class Poisson(Distribution):

    def __init__(self, distribution):
        super().__init__(
            name=distribution["name"]
        )
        self.lambdaa = distribution["lambda"]

    def generate_points(self, number_of_events, problem_dimension):
        """
        This function generates datapoints using Poisson distribution
        """

        # initialize vector with required dimension
        points = np.zeros((number_of_events, problem_dimension))

        # loop over problem dimension to generate each dimension
        for i in range(0, problem_dimension):
            points[:, i] = np.array(np.random.poisson(self.lambdaa[i], number_of_events))

        return points


# ================================
# Exponential Distribution Class
# ================================
class Exponential(Distribution):

    def __init__(self, distribution):
        super().__init__(
            name=distribution["name"]
        )
        self.lambdaa = distribution["lambda"]

    def generate_points(self, number_of_events, problem_dimension):
        """
        This function generates datapoints using Exponential distribution
        """

        # initialize vector with required dimension
        points = np.zeros((number_of_events, problem_dimension))

        # loop over problem dimension to generate each dimension
        for i in range(0, problem_dimension):
            points[:, i] = np.array(np.random.exponential(self.lambdaa[i], number_of_events))

        return points


# ================================
# Gamma Distribution Class
# ================================
class Gamma(Distribution):
    def __init__(self, distribution):

        """
        name: name of the distribution
        k : k parameter of gamma distribution
        tau :  parameter of gamma distribution
        """

        super().__init__(
            name=distribution["name"]
        )
        self.k = distribution["k"]
        self.tau = distribution["tau"]

    def generate_points(self, number_of_events, problem_dimension):
        """
        This function generates datapoints using Gamma distribution
        """

        # initialize vector with required dimension
        points = np.zeros((number_of_events, problem_dimension))

        # loop over problem dimension to generate each dimension
        for i in range(0, problem_dimension):
            points[:, i] = np.array(np.random.gamma(self.k, self.tau, number_of_events))
        return points


#================================
# Gaussian_Gamma Distribution Class
#================================
class Gaussian_Gamma(Distribution):

    def __init__(self, distribution):
        super().__init__(
            name = distribution["name"]
        )
        self.cut = distribution["cut"]
        self.distributions_params = distribution["distributions_params"]

    def generate_points(self, number_of_events, problem_dimension):
        """
        This function generates datapoints using Gamma distribution
        """

        # initialize vector with required dimension
        points = np.zeros((number_of_events, problem_dimension))
        
        # loop over problem dimension to generate each dimension
        for i in range(0, problem_dimension) :
            dimension_params = self.distributions_params[i]
            distrib_type = dimension_params["distrib"]
            distrib_param_1, distrib_param_2 = dimension_params["param_1"], dimension_params["param_2"]
            if len(self.cut[i]) == 0:
                if distrib_type == DISTRIBUTION_GAMMA :
                    k,tau = distrib_param_1,distrib_param_2
                    points[:, i] = np.array(np.random.gamma(k, tau, number_of_events))
                elif distrib_type == DISTRIBUTION_GAUSSIAN :
                    mu, sigma = distrib_param_1,distrib_param_2
                    points[:, i] = np.array(np.random.normal(mu,sigma, number_of_events))
            else :
                # get min and max limit of the cut
                min_lim, max_lim = self.cut[i]
                points_i = np.array([])
                # loop over points until points are equial to number of events
                while len(points_i) < number_of_events:
                    # generate points
                    if distrib_type == DISTRIBUTION_GAMMA :
                        k,tau = distrib_param_1,distrib_param_2
                        points_generated = np.array(np.random.gamma(k, tau, number_of_events))
                    elif distrib_type == DISTRIBUTION_GAUSSIAN :
                        mu, sigma = distrib_param_1,distrib_param_2
                        points_generated = np.array(np.random.normal(mu, sigma, number_of_events))                    # remove points not in limits
                    points_generated = points_generated [ (points_generated >=min_lim) * (points_generated <= max_lim)]
                    # append points to previously generated points
                    points_i  = np.append(points_i, points_generated)
                    # remove points if more than number of events
                    if len(points_i) > number_of_events:
                        points_i = points_i[:number_of_events]
                
                points[:, i] = points_i
        return points
        