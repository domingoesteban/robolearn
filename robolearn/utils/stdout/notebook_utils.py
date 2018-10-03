from IPython import get_ipython


def is_ipython():
    return type(get_ipython()).__module__.startswith('ipykernel.')
