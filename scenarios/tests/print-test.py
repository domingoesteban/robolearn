import time
from robolearn.utils.print_utils import ProgressBar

max_value = 67
title = "Dummy bar"

bar = ProgressBar(max_value, bar_title=title)

for i in range(max_value):
    time.sleep(0.1)
    bar.update(i)

bar.end()
print('')