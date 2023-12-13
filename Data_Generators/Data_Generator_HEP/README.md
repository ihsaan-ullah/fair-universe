# Data Generator HEP

This folder contains the data generator (`datagen_hep.py`) to generate HEP Competition datasets. and a data simulator (`data_simulator.py`) which lets you generate different datasets with and without bootstraping. 

To run the `datagen_hep.py` script use the following code
```
python datagen_hep.py <input_file> <crosssection_json_file> <output_file_location>

```
In case you didn't specify the output file the out will be in `fair-universe/Data_Generators/Full_Dataset/' 

This will generate train and test sets. 

The `data_simulator.py` is a module and hence can be imported into your codes. 