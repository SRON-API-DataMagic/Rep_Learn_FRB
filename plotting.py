import matplotlib.pyplot as plt
import numpy as np

def plot_dynamic_spectra(dynamic_spectra, title="Dynamic Spectra"):
    """
    Plot a dynamic spectrum with horizontal time axis and reversed frequency axis.

    Parameters:
        dynamic_spectra (numpy.ndarray): The dynamic spectrum to plot.
        title (str): The title for the plot (default is "Dynamic Spectra").

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(dynamic_spectra.T, aspect="auto", cmap="viridis")
    plt.xlabel("Time Sample #", size=14)
    plt.ylabel("Frequency", size=14)
    plt.colorbar()
    plt.title(title, size=16)
    plt.show()

def plot_lightcurve(time_samples, lightcurve):
    """
    Plot a lightcurve.

    Parameters:
        time_samples (numpy.ndarray): Array of time sample numbers.
        lightcurve (numpy.ndarray): Array of intensity values.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time_samples, lightcurve, color='blue', lw=1)
    plt.xlabel("Time Sample #", fontsize=12)
    plt.ylabel("Intensity", fontsize=12)
    plt.title("FRB Lightcurve", fontsize=14)
    plt.grid(True)
    plt.show()