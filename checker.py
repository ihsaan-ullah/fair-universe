#================================
# Checker Class
#================================
class Checker:

    def settings_is_not_loaded(self, settings):
        return (settings is None)

    def distributions_are_not_loaded(self, params_distributions):
        return (not params_distributions)

    def systematics_are_not_loaded(self, params_systematics):
        return (params_systematics  is None)

    def data_is_not_generated(self, generated_dataframe):
        return (generated_dataframe  is None)
