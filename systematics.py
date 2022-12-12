#================================
# Imports
#================================
import numpy as np

#================================
# Benjamin 
#================================
class Ben:
    def __init__(self, systematics):

        # name: name of the systematic 
        self.name = systematics["name"]

        # allowed_dimension : dimension of data required
        self.allowed_dimension = systematics["allowed_dimension"]

        # number_of_nuissance_values : number of nuissance values required
        self.number_of_nuissance_values = systematics["number_of_nuissance_values"]


#================================
# Translation 
#================================
class Translation:
    def __init__(self, systematics):

        # name: name of the systematic 
        self.name = systematics["name"]

        # allowed_dimension : dimension of data required
        self.allowed_dimension = systematics["allowed_dimension"]


#================================
# Scaling 
#================================
class Scaling:
    def __init__(self, systematics):

        # name: name of the systematic 
        self.name = systematics["name"]

        # allowed_dimension : dimension of data required
        self.allowed_dimension = systematics["allowed_dimension"]