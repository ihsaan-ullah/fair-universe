#================================
# Imports
#================================
import numpy as np


#================================
# Systematics Class  
#================================
class Systematics:
    
    def __init__(self, name=None, allowed_dimension=None, number_of_nuissance_values=None):
        
        """
        name: name of the systematic 
        allowed_dimension : dimension of data required
        number_of_nuissance_values : number of nuissance values required
        """

        # init parameters
        self.name = name
        self.allowed_dimension = allowed_dimension
        self.number_of_nuissance_values = number_of_nuissance_values


#================================
# Benjamin  
#================================
class Ben(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name = systematics["name"],
            allowed_dimension = systematics["allowed_dimension"],
            number_of_nuissance_values = systematics["number_of_nuissance_values"]
        )



#================================
# Translation 
#================================
class Translation(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name = systematics["name"],
            allowed_dimension = systematics["allowed_dimension"]
        )


#================================
# Scaling 
#================================
class Scaling(Systematics):
    def __init__(self, systematics):

        super().__init__(
            name = systematics["name"],
            allowed_dimension = systematics["allowed_dimension"]
        )