def is_iterable(x):
    """Check if variable is iterable
    """
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True
