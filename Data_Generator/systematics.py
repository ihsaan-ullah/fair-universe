# ================================
# Imports
# ================================
import numpy as np
from math import radians
from logger import Logger


# ================================
# Systematics Class
# ================================
class Systematics:
    def __init__(self, name=None, allowed_dimension=None):

        """
        name: name of the systematic 
        allowed_dimension : dimension of data required
        number_of_nuissance_values : number of nuissance values required
        """

        # init parameters
        self.name = name
        self.allowed_dimension = allowed_dimension
        self.logger = Logger()


# ================================
# Benjamin
# ================================
class Ben_New(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name=systematics["name"],
            allowed_dimension=systematics["allowed_dimension"]
        )

        self.sigma_bias = systematics["sigma_bias"]
        self.mu_bias = systematics["mu_bias"]

    def apply_systematics(self, problem_dimension, points):

        if problem_dimension != self.allowed_dimension:
            self.logger.error("problem_dimension and allowed_dimension must be same for this systematics!")
            exit()

        # apply translation
        updated_points = (points + self.mu_bias)

        # apply scaling
        updated_points = (updated_points * self.sigma_bias)

        return updated_points


# ================================
# Translation
# ================================
class Translation(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name=systematics["name"],
            allowed_dimension=systematics["allowed_dimension"]
        )
        self.translation_vector = np.array(systematics["translation_vector"])

    def apply_systematics(self, problem_dimension, points):

        if len(self.translation_vector) != problem_dimension:
            self.logger.error("translation_vector has invalid length!")
            exit()
        return (points + self.translation_vector)


# ================================
# Scaling
# ================================
class Scaling(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name=systematics["name"],
            allowed_dimension=systematics["allowed_dimension"]
        )
        self.scaling_vector = np.array(systematics["scaling_vector"])

    def apply_systematics(self, problem_dimension, points):

        if len(self.scaling_vector) != problem_dimension:
            self.logger.error("scaling_vector has invalid length!")
            exit()

        return (points * self.scaling_vector)


# ================================
# Box
# ================================
class Box(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name=systematics["name"]
        )
        self.box_center = systematics["box_center"]
        self.box_length = systematics["box_length"]

    def apply_systematics(self, df_org, df_bia):

        box_x = [self.box_center[0]-self.box_length, self.box_center[0]+self.box_length]
        box_y = [self.box_center[1]-self.box_length, self.box_center[1]+self.box_length]

        mask_org = (df_org['x1'] >= box_x[0]) & (df_org['x1'] <= box_x[1]) & (df_org['x2'] >= box_y[0]) & (df_org['x2'] <= box_y[1])
        mask_bia = (df_bia['x1'] >= box_x[0]) & (df_bia['x1'] <= box_x[1]) & (df_bia['x2'] >= box_y[0]) & (df_bia['x2'] <= box_y[1])

        df_org = df_org.loc[mask_org]
        df_bia = df_bia.loc[mask_bia]

        df_org.reset_index(inplace=True)
        df_bia.reset_index(inplace=True)

        return df_org[["x1", "x2", "y"]], df_bia[["x1", "x2", "y"]]


# ================================
# Rotation
# ================================
class Rotation(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name=systematics["name"]
        )
        self.rotation_degree = radians(systematics["rotation_degree"])
        # systematics["rotation_degree"]*np.pi/180  # In degreess
        self.rotation_matrix = np.array([
            [np.cos(self.rotation_degree), -np.sin(self.rotation_degree)],
            [np.sin(self.rotation_degree), np.cos(self.rotation_degree)]
        ])

    def apply_systematics(self, problem_dimension, points):
        return (np.matmul(self.rotation_matrix, points.T).T)
