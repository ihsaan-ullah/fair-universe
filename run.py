from data_generator import DataGenerator

data_gen = DataGenerator()

data_gen.load_settings()
data_gen.load_distributions()
data_gen.load_systematics()
data_gen.generate_data()
data_gen.get_data()
data_gen.show_statistics()
data_gen.visulaize_data()
data_gen.visualize_distributions()
