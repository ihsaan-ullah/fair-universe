# Fair Universe Data Generation
This repository shows how data is generated for the toy problem
***

## Run Code
To run the code with default settings, use the following code:

```python
# import data generator
from data_generator import DataGenerator

# initialize data generator instance
data_gen = DataGenerator()

# load settings from JSON
data_gen.load_settings()

# load distributions from settings
data_gen.load_distributions()

# load systematics from settings
data_gen.load_systematics()

# generate data using distributions and systematics
data_gen.generate_data(apply_systematics=True)

# get data as a dataframe
df = data_gen.get_data()

# show statistics of data
data_gen.show_statistics()

# show distribution parameters
data_gen.show_distribution_parameters()

# show distribution parameters
data_gen.show_systematics_parameters()

# visualize data
data_gen.visualize_data()

# visualize data distribution (1D histogram)
data_gen.visualize_distributions_1d()

# visualize data distribution (2D histogram)
data_gen.visualize_distributions_2d()

# save data as csv
data_gen.save_data()

```
***

The following documentation describes the components and working of the whole pipleine of data generation. 

***


## Settings
`settings.json` file consists of all the settings required for data generation:

**1- Dimension of problem**
```json
"problem_dimension" : 2
```

**2- Number of Events**
```json
"number_of_events" : 1000
```

**3- signal distribution**
```json
"signal_distribution": {
    "name": "Gaussian",
    "mu" : [2,2],
    "sigma" : [1, 1],
    "cut" : [[2,4], []]
},
```
**4- background distribution**
```json
"background_distribution": {
    "name": "Gaussian",
    "mu" : [6,6],
    "sigma" : [1, 1],
    "cut" : [[6,8], []]
},
```

**5- systematics**
```json
"systematics" : {
    "name" : "Ben_New",
    "allowed_dimension" : 2,
    "sigma_bias" : [1,1],
    "mu_bias" : [1,1]
}
```

## Distributions
`distributions.py` file contains separate classes for different distributions to be used for signal or background. Currently supported distributions are:

**1- Gaussian**
```json
{
    "name": "Gaussian",
    "mu" : [2,2],
    "sigma" : [1, 1],
    "cut" : [[2,4], []]
}
```
**2- Exponential**
```json
{
    "name": "Exponential",
    "lambda" : [3,3],
    "cut" : [[], []]
}
```
**3- Poisson**
```json
{
    "name": "Exponential",
    "lambda" : [500,500],
    "cut" : [[], []]
}
```

## Systematics
`systematics.py` file consists of classes for different systematics to be added to the data to introduce bias in the data. Currently supported systematics are:

**1- Ben's systematics (for 2D problem only)**
```json
{
    "name" : "Ben_New",
    "allowed_dimension" : 2,
    "sigma_bias" : [1,1],
    "mu_bias" : [1,1]
}
```
**2- Translation**
```json
{
    "name" : "Translation",
    "allowed_dimension" : -1
    "translation_vector" : [5,2]
}
```
**3- Scaling**
```json
{
    "name" : "Scaling",
    "allowed_dimension" : -1
    "scaling_vector" : [5,2]
}
```

## Logger
`logger.py` consists of a class which is responsible for showing warning, error and success messages.


## Checker
`checker.py` consists of a class which is responsible for checking data members at each step of data generation.


## Data Generation
`data_generatior.py` file is the entry point to generate data with systematics. This file consists of ***DataGenerator*** class with the following methods and functionalities:

**1- Load settings** (`load_settings`)  
Loads `settings.json` file 

**2- Load distributions** (`load_distributions`)  
sets up *signal* and *background* distributions from settings

**3- Load systematics** (`load_systematics`)  
sets up systematics from settings to be applied to data generaton

**4- Generate data** (`generate_data`)  
generates data according to the settings (distributions and systematics) with option to choose if systematics should be applied or not.

**5- Get data** (`get_data`)  
returns the generated data as *Pandas DataFrame*

**6- Show statistics** (`show_statistics`)  
shows statistics of the data: data dimension, number of examples, number of classes, class labels etc.

**7- Show distribution parameters** (`show_distribution_parameters`)  
shows signal and background distributions parameters

**8- Show systematics parameters** (`show_systematics_parameters`)  
shows systematics parameters parameters

**9- Visualize data** (`visulaize_data`)   
visulaizes the generated data

**10- Visualize distributions** (`visualize_distributions`)   
visulaizes the histogram of generated data

**11- Save data** (`save_data`)  
saves the generated data in a csv file.