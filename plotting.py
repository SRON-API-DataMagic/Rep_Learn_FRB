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
    # Calculate the time step based on the number of time samples
    num_time_samples = dynamic_spectra.shape[0]
    time_step = 0.0000256  # Default time step in seconds

    # Calculate the extent based on the number of frequency channels
    extent = [0, num_time_samples * time_step * 1000, 1720, 1208]

    plt.figure(figsize=(10, 6))
    plt.imshow(dynamic_spectra.T, aspect="auto", cmap="viridis", extent=extent)
    plt.xlabel("Time (ms)", size=14)
    plt.ylabel("Frequency (MHz)", size=14)
    plt.gca().invert_yaxis()
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
    # Calculate the time values based on the time step
    time_values = time_samples * 0.0000256 * 1000  # Convert to milliseconds

    plt.figure(figsize=(12, 6))
    plt.plot(time_values, lightcurve, color='blue', lw=1)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Intensity", fontsize=12)
    plt.title("FRB Lightcurve", fontsize=14)
    plt.grid(True)
    plt.show()