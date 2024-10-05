import argparse

# New type for argparse
def positiveint(value: str) -> int:

    """Type function for argparse - Positive int"""

    try:
        intvalue = int(value)
    except:
        raise argparse.ArgumentTypeError (f"{value} is not a valid positive int")

    if intvalue <= 0:
        raise argparse.ArgumentTypeError (f"{value} is not a valid positive int")

    return intvalue

# New type for argparse
def floatrange(lower_bound: float, upper_bound: float) -> float:

    """
        Type function for argparse - Float with bounds

         lower_bound - minimum acceptable parameter
         upper_bound - maximum acceptable parameter
    """

    # Define the function with default arguments
    def float_range_checker(value: str) -> float:

        try:
            f = float(value)
        except:
            raise argparse.ArgumentTypeError(f"{value} is not a valid float")

        if f < lower_bound or f > upper_bound:
            raise argparse.ArgumentTypeError(f"Argument must be in range [{lower_bound},{upper_bound}]")
        
        return f

    return float_range_checker