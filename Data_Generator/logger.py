#================================
# Logger Class
#================================
class Logger:
    def __init__(self, show_logs=True):
        self._warning_sign = "[!]"
        self._error_sign = "[-]"
        self._success_sign = "[+]"
        self.show_logs = show_logs

    def warning(self, msg):
        if self.show_logs:
            self.print_line()
            print(self._warning_sign, msg)
            self.print_line()

    def error(self, msg):
        self.print_line()
        print(self._error_sign, msg)
        self.print_line()

    def success(self, msg):
        if self.show_logs:
            self.print_line()
            print(self._success_sign, msg)
            self.print_line()


    def print_line(self):
        if self.show_logs:
            print("-----------------------------------")