import sys


class PrintColor(object):
    def __init__(self):
        self.colors = {
            'WHITE': "\033[1;37m",
            'RED': "\033[1;31m",
            'GREEN': "\033[1;32m",
            'YELLOW': "\033[1;33m",
            'BLUE': "\033[1;34m",
            'MAGENTA': "\033[1;35m",
            'CYAN': "\033[1;36m",
            'GRAY': "\033[1;37m",
            'PURPLE': "\033[1;57m",
            'RESET': "\033[0;0m",
            'BOLD': "\033[;1m",
            'REVERSE': "\033[;7m",
        }

    def change(self, color):
        if color.upper() not in self.colors.keys():
            raise ValueError("Wrong color!!")
        sys.stdout.write(self.colors[color.upper()])

    def reset(self):
        sys.stdout.write(self.colors['RESET'])
