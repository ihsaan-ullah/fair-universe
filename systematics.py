#================================
# Imports
#================================
import numpy as np
from logger import Logger


#================================
# Systematics Class  
#================================
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
        


#================================
# Benjamin  
#================================
class Ben_New(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name = systematics["name"],
            allowed_dimension = systematics["allowed_dimension"]
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
        
         



#================================
# Translation 
#================================
class Translation(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name = systematics["name"],
            allowed_dimension = systematics["allowed_dimension"]
        )
        self.translation_vector = np.array(systematics["translation_vector"])
        

    def apply_systematics(self, problem_dimension, points):
        
        if len(self.translation_vector) != problem_dimension:
            self.logger.error("translation_vector has invalid length!")
            exit()
        return (points + self.translation_vector)



#================================
# Scaling 
#================================
class Scaling(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name = systematics["name"],
            allowed_dimension = systematics["allowed_dimension"]
        )
        self.scaling_vector = np.array(systematics["scaling_vector"])

    def apply_systematics(self, problem_dimension, points):
        
        if len(self.scaling_vector) != problem_dimension:
            self.logger.error("scaling_vector has invalid length!")
            exit()

        return (points * self.scaling_vector)