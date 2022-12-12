#================================
# Imports
#================================
import numpy as np

#================================
# Gaussian Distribution Class
#================================
class Gaussian:
    def __init__(self, distribution, problem_dimension):

        # name: name of the distribution 
        self.name == distribution["name"]

        # mu : 𝜇 parameter of gaussian distribution
        self.mu = distribution["mu"]

        # sigma : 𝜎 parameter of gaussian distribution
        self.sigma = distribution["sigma"]

        # problem_dimension: dimension of generated data
        self.problem_dimension = problem_dimension

    def get_points(self, number_of_events):
        """
        This function generates datapoints using Gaussian distribution
        """
        return np.random.normal(self.mu,self.sigma,number_of_events)

    def pdf_1d(self, x):
        """
        This function generates 1D gaussian PDF

        Args:
        -----
        x : array of 1D input

        Returns:
        --------
        gaussian PDF
        
        Latex:
        ------
        \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
        """
        return (1/(self.sigma*(2*np.pi)**0.5))*np.exp((-1/2)*((x-self.mu)/self.sigma)**2)


#================================
# Poisson Distribution Class
#================================
class Poisson:
    
    def __init__(self, distribution, problem_dimension):

        # name: name of the distribution 
        self.name == distribution["name"]

        # lambdaa : 𝜆 parameter of poisson distribution
        self.lambdaa = distribution["lambda"]

        # problem_dimension: dimension of generated data
        self.problem_dimension = problem_dimension

        
    def get_points(self, number_of_events):
        """
        This function generates datapoints using Poisson distribution
        """
        return np.random.poisson(self.lambdaa, number_of_events)

  
#================================
# Exponential Distribution Class
#================================
class Exponential:
    def __init__(self, distribution, problem_dimension):

        # name: name of the distribution 
        self.name == distribution["name"]

        # lambdaa : 𝜆 parameter of exponential distribution
        self.lambdaa = distribution["lambda"]

        # problem_dimension: dimension of generated data
        self.problem_dimension = problem_dimension

    def get_points(self, number_of_events):
        """
        This function generates datapoints using Exponential distribution
        """
        return np.random.exponential(self.lambdaa,number_of_events)

    def pdf_1d(self, x):
        """
        This function generates 1D exponential PDF

        Args:
        -----
        x : array of 1D input
        

        Returns:
        --------
        exponential PDF

        Latex:
        ------
        \begin{cases}
        \lambda e^{-\lambda x} & x\ge 0\\
        0 & x < 0
        \end{cases}  
        """

        return self.lambdaa*np.exp(-self.lambdaa*x)
    
