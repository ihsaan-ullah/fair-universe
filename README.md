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
data_gen.generate_data()

# get data as a dataframe
data_gen.get_data()

# show statistics of data
data_gen.show_statistics()

# visualize data
data_gen.visulaize_data()

```

or run `run.py` file 
```bash
python run.py
```
***

The following documentation describes the components and working of the whole pipleine of data generation. 

***


## Settings
`settings.json` file consists of all the settings required for data generation:

**1- signal distribution**
```json
"signal_distribution": {
    "name": "Gaussian",
    "mu" : 3,
    "sigma" : 0.8,
    "number_of_events" : 1000
}
```
**2- background distribution**
```json
"background_distribution": {
    "name": "Exponential",
    "lambda" : 3,
    "number_of_events" : 1000
}
```

**3- Dimension of problem**
```json
"problem_dimension" : 2
```
**4- systematics**
```json
"systematics" : {
"name" : "ben",
"allowed_dimension" : 2,
"number_of_nuissance_values" : 2
}
```

## Distributions
`distributions.py` file contains separate classes for different distributions to be used for signal or background. Currently supported distributions are:

**1- Gaussian**
```json
{
    "name": "Gaussian",
    "mu" : 3,
    "sigma" : 0.8,
    "number_of_events" : 1000
}
```
**2- Exponential**
```json
{
    "name": "Exponential",
    "lambda" : 3,
    "number_of_events" : 1000
}
```
**3- Poisson**
```json
{
    "name": "Poisson",
    "lambda" : 500,
    "number_of_events" : 1000
}
```

## Systematics
`systematics.py` file consists of classes for different systematics to be added to the data to introduce bias in the data. Currently supported systematics are:

**1- Ben's systematics (for 2D problem only)**
```json
{
    "name" : "ben",
    "allowed_dimension" : 2,
    "number_of_nuissance_values" : 2
}
```
**2- Translation**
```json
{
    "name" : "translation",
    "allowed_dimension" : -1
}
```
**3- Scaling**
```json
{
    "name" : "scaling",
    "allowed_dimension" : -1
}
```

## Errors
`errors.py` consists of a class which is responsible for showing errors and warnings.


## Data Generation
`data_generatior.py` file is the entry point to generate data with systematics. This file consists of ***DataGenerator*** class with the following methods and functionalities:

**1- Load settings** (`load_settings`)  
Loads `settings.json` file 

**2- Load distributions** (`load_distributions`)  
sets up *signal* and *background* distributions from settings

**3- Load systematics** (`load_systematics`)  
sets up systematics from settings to be applied to data generaton

**4- Generate data** (`generate_data`)  
generates data according to the settings (distributions and systematics)

**5- Get data** (`get_data`)  
returns the generated data as *Pandas DataFrame*

**6- Show statistics** (`show_statistics`)  
shows statistics of the data: data dimension, number of examples, number of classes, class labels etc.

**7- Visualize data** (`visulaize_data`)   
visulaizes the generated data