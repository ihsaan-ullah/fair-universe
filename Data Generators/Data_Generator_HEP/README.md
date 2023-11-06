# Data Generator HEP

This folder contains data generator to generate HEP datasets.

To run the scrip put `reference_data.csv` in the same folder i.e. `Data_Generator_HEP` and run the following command


```
python datagen_hep.py

```

This will generate train and test sets. Note that the test sets are combined as sets of 100 test sets. E.g. `set_0` is a folder with 100 test sets with different values of systematics and one value of $\mu$. 