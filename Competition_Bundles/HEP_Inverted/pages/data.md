# Data
***

### ⬇️ [Download Data Generator](https://www.codabench.org/datasets/download/a3d2684b-697b-453c-b2df-fce3f359d8f7/)
The data used in this competition is generated by a Data Generator. This data generator can be used to generate datasets with 2D features. 

### How to use Data Generator
You can use the data generator to generate datasets with or without systematics. The following code show the usage of the data generator

**Import Data Generator**  
```
# import data generator
from Data_Generator.data_generator_physics import DataGenerator
```

**Define Systematics**  
```
# initialize systematics
systematics = [{
        "name": "Translation",
        "z_range": [-10, 10],
        "z_angles": [45]
}]

# if you don't want to use systematics, initialize systematics with None
# systematics = None
```

**Define Params**  
```
#define params dict
params = {
        "pi": 0.1,
        "nu_1: 100000,
        "mu_range": [0.9, 1.1],
        "systematics": systematics,
}

# To use just one value of mu e.g. 1.0 , replace mu_range with the following
# "mu_range": [1.0, 1.0],
```

**Generate Dataset**  
```
data_gen = DataGenerator(params=params, SEED=33)
data_gen.generate_data()
generated_dataset = data_gen.get_data()
```