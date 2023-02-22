#================================
# Logger Class
#================================
class Logger:
    def __init__(self):
        self._warning_sign = "[!]"
        self._error_sign = "[-]"
        self._success_sign = "[+]"

    def warning(self, msg):
        self.print_line()
        print(self._warning_sign, msg)
        self.print_line()

    def error(self, msg):
        self.print_line()
        print(self._error_sign, msg)
        self.print_line()

    def success(self, msg):
        self.print_line()
        print(self._success_sign, msg)
        self.print_line()


    def print_line(self):
        print("-----------------------------------")