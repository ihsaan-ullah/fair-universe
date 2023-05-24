# ================================
# Imports
# ================================
import numpy as np
from constants import (
    GAUSSIAN_GENERETOR_TYPE_NORMAL,
    GAUSSIAN_GENERETOR_TYPE_MULTIVARIATE
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
                [np.cos(np.radians(self.angle_rotation)), -np.sin(np.radians(self.angle_rotation))],
                [np.sin(np.radians(self.angle_rotation)), np.cos(np.radians(self.angle_rotation))]
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
        _theta_ :θ parameter of gamma distribution
        """

        super().__init__(
            name=distribution["name"]
        )
        self.k = distribution["k"]
        self._theta_ = distribution["_theta_"]

    def generate_points(self, number_of_events, problem_dimension):
        """
        This function generates datapoints using Gamma distribution
        """

        # initialize vector with required dimension
        points = np.zeros((number_of_events, problem_dimension))

        # loop over problem dimension to generate each dimension
        for i in range(0, problem_dimension):
            points[:, i] = np.array(np.random.gamma(self.k[i], self._theta_[i], number_of_events))
        return points
