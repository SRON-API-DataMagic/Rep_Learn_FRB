import tempfile
from urllib import request

import matplotlib.pyplot as plt
import numpy as np
from jess.dispersion import dedisperse
from jess.fitters import median_fitter
from scipy.stats import median_abs_deviation
from will import create, inject, detect
from your import Your
from your.formats.filwriter import make_sigproc_object


def create_filterbank_with_noise(output_file, num_time_samples=8192, num_frequency_channels=4096,
                                 std_value=1, mean_value=0):
    """
    Create a filterbank file with synthetic Gaussian noise.

    Parameters:
    - output_file (str): The output filterbank file name.
    - num_time_samples (int): Number of time samples in the filterbank data. Default is 8192.
    - num_frequency_channels (int): Number of frequency channels in the filterbank data. Default is 4096.
    - std_value (float): The standard deviation of the generated Gaussian noise. Default is 1.
    - mean_value (float): The mean value of the generated Gaussian noise. Default is 0.

    Returns:
    None
    """

    # Generate Gaussian noise with the desired standard deviation and mean
    noise = np.random.normal(mean_value, std_value, size=(num_time_samples, num_frequency_channels))

    # Make sure datatype is correct (float32)
    noise = noise.astype(np.float32)

    # Create a SIGPROC filterbank object with specified header parameters
    sigproc_object = make_sigproc_object(
        rawdatafile=output_file,
        source_name="TEMP",
        nchans=num_frequency_channels,
        foff=-0.234375,
        fch1=1919.8828125,
        tsamp=0.000256,
        tstart=59319.97462321287,
        src_raj=112233.44,
        src_dej=112233.44,
        machine_id=0,
        nbeams=0,
        ibeam=0,
        nbits=32,  # Changed to 32 bits per data sample to match the float32 data type
        nifs=1,
        barycentric=0,
        pulsarcentric=0,
        telescope_id=6,
        data_type=1,  # Changed to 1 for float32 data type
        az_start=-1,
        za_start=-1,
    )

    # Write the header information to the filterbank file
    sigproc_object.write_header(output_file)

    # Append the data to the filterbank file
    sigproc_object.append_spectra(noise, output_file)


def inject_pulse_into_dynamic_spectrum(dynamic_spectra, pulse, pulse_start_time=None):
    """
    Inject a pulse into a dynamic spectrum.

    Parameters:
    - dynamic_spectra (numpy.ndarray): The dynamic spectrum into which the pulse will be injected.
    - pulse (numpy.ndarray): The pulse to inject into the dynamic spectrum.
    - pulse_start_time (int): The time sample where you want to inject the pulse. If None, it will be placed in the
                              middle of the dynamic spectrum.

    Returns:
    - numpy.ndarray: The dynamic spectrum with the injected pulse.
    """

    # Make a copy of the dynamic spectrum
    dynamic_spectra_copy = dynamic_spectra.copy()

    # Define the time sample where you want to inject the pulse, for now middle of spectrum
    if pulse_start_time is None:
        pulse_start_time = dynamic_spectra_copy.shape[0] // 2  # Adjust this to control the injection time

    # Ensure the dimensions of pulse match the target region
    desired_shape = dynamic_spectra_copy[
        pulse_start_time:pulse_start_time + dynamic_spectra_copy.shape[0], :].shape
    pulse_resized = np.resize(pulse, desired_shape)

    # Inject the resized pulse into the copied dynamic spectrum
    dynamic_spectra_copy[pulse_start_time:pulse_start_time + dynamic_spectra_copy.shape[0], :] += pulse_resized

    return dynamic_spectra_copy

if __name__ == "__main__":
    # Inject the pulse into the dynamic spectrum
    dynamic_spectra = [0]
    pulse = [0]
    dynamic_spectra_with_pulse = inject_pulse_into_dynamic_spectrum(dynamic_spectra, pulse)