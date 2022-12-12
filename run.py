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