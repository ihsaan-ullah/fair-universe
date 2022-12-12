#================================
# Errors Class
#================================
class Errors:
    def __init__(self):
        self._warning_sign = "[!]"
        self._error_sign = "[-]"

    def warning(self, msg):
        
        print(self._warning_sign, msg)

    def error(self, msg):

        print(self._error_sign, msg)
