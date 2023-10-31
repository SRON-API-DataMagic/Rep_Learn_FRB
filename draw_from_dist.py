import numpy as np

def generate_center_freq(min_freq, max_freq):
    """
    Generate a random frequency center within the specified range.

    Parameters:
        min_freq (float): Minimum frequency in MHz.
        max_freq (float): Maximum frequency in MHz.

    Returns:
        float: Random center frequency within the specified range.
    """
    center_freq = np.random.uniform(min_freq, max_freq)
    return center_freq

def generate_sigma_freq(min_sigma, max_sigma):
    """
    Generate a random frequency width (sigma_freq) within the specified range.

    Parameters:
        min_sigma (float): Minimum frequency width (sigma_freq).
        max_sigma (float): Maximum frequency width (sigma_freq).

    Returns:
        float: Random sigma_freq within the specified range.
    """
    sigma_freq = np.random.uniform(min_sigma, max_sigma)
    return sigma_freq

def generate_sigma_time(mean, std):
    """
    Generate a random time width (sigma_time) from a normal distribution within the specified range.

    Parameters:
        mean (float): Mean value of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        float: Random sigma_time within the specified range.
    """
    sigma_time = np.random.normal(loc=mean, scale=std)
    return sigma_time
