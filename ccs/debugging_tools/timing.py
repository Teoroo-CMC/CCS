from functools import wraps
from time import time


def timing(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        print(f"func {func.__name__} took {time_end-time_start}")
        return result

    return wrap
