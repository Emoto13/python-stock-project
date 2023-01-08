from functools import wraps


def try_catch(func):
    """ Wraps the decorated function in a try-catch.
     If function fails return the exception. """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except Exception as exception:
            print(f"Exception in {func.__name__}: {exception}")
            return exception
    return wrapper
