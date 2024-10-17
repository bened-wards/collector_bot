
def check_close(a, b, percentage_tolerance=10):
    bigger = a if a > b else b
    smaller = a if a < b else b
    return bigger * (1 - percentage_tolerance / 100) < smaller < bigger * (1 + percentage_tolerance / 100)
