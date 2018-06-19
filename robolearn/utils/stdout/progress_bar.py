import sys


class ProgressBar(object):
    def __init__(self, max_val, total_lines=50, bar_title=None, bar_symbol='#'):
        self.max_val = max_val
        self.total_lines = total_lines
        self.bar_symbol = bar_symbol

        if bar_title is None:
            bar_title = ''

        sys.stdout.write(bar_title + ": [" + "-" * (self.total_lines-1) + "]" +
                         chr(8) * self.total_lines)
        sys.stdout.flush()
        self.progress = 0

    def update(self, i):
        x = int(i * self.total_lines // self.max_val)
        sys.stdout.write(self.bar_symbol * (x - self.progress))
        sys.stdout.flush()
        self.progress = x

    def end(self):
        sys.stdout.write("#" * (self.total_lines - self.progress - 1) + "]\n")
        sys.stdout.flush()
