#================================
# Logger Class
#================================
class Logger:
    def __init__(self):
        self._warning_sign = "[!]"
        self._error_sign = "[-]"
        self._success_sign = "[+]"

    def warning(self, msg):
        print(self._warning_sign, msg)

    def error(self, msg):
        print(self._error_sign, msg)

    def success(self, msg):
        print(self._success_sign, msg)
