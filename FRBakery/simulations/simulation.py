import tempfile
import matplotlib.pyplot as plt
import numpy as np
from jess.dispersion import dedisperse
from jess.fitters import median_fitter
from scipy.stats import median_abs_deviation
from will import create, inject, detect
from your import Your
from your.formats.filwriter import make_sigproc_object
from scipy.signal import convolve


def create_filterbank_with_noise(output_file, num_time_samples=8192, num_frequency_channels=4096, channel_start=2000,
                                 std_value=1, mean_value=0):
    """
    Create a filterbank file with synthetic Gaussian noise.

    Parameters:
    - output_file (str): The output filterbank file name.
    - num_time_samples (int): Number of time samples in the filterbank data. Default is 8192.
    - num_frequency_channels (int): Number of frequency channels in the filterbank data. Default is 4096.
    - channel_start (float): Frequency of first channel (MHz)
    - std_value (float): The standard deviation of the generated Gaussian noise. Default is 1.
    - mean_value (float): The mean value of the generated Gaussian noise. Default is 0.

    Returns:
    None
    """
    
    # Generate synthetic Gaussian noise
    noise_data = np.random.normal(loc=mean_value, scale=std_value, size=(num_time_samples, num_frequency_channels)).astype(np.float32)

    # Create a SIGPROC filterbank object with specified header parameters
    sigproc_object = make_sigproc_object(
        rawdatafile=output_file,
        source_name="TEMP",
        nchans=num_frequency_channels,
        foff=-1,
        fch1=channel_start,
        tsamp=81.92e-6,
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

    # Append the noise data to the filterbank file
    sigproc_object.append_spectra(noise_data, output_file)


def inject_pulse_into_dynamic_spectrum(dynamic_spectra, pulse, pulse_start_time=None):
    """
    Inject a pulse into a dynamic spectrum.

    Parameters:
    - dynamic_spectra (numpy.ndarray): The dynamic spectrum into which the pulse will be injected.
    - pulse (numpy.ndarray): The pulse to inject into the dynamic spectrum.
    - pulse_start_time (int, optional): The time sample where you want to inject the pulse.
      If None (default), it will be placed in the middle of the dynamic spectrum.

    Returns:
    - numpy.ndarray: The dynamic spectrum with the injected pulse.
    """

    # Make a copy of the dynamic spectrum
    dynamic_spectra_copy = dynamic_spectra.copy()

    # Calculate the time sample where you want to inject the pulse
    if pulse_start_time is None:
        pulse_start_time = dynamic_spectra_copy.shape[0] // 2 - pulse.shape[0] // 2

    # Ensure the dimensions of the pulse match the target region
    desired_shape = dynamic_spectra_copy[pulse_start_time:pulse_start_time + pulse.shape[0], :].shape
    pulse_resized = np.resize(pulse, desired_shape)

    # Inject the resized pulse into the copied dynamic spectrum
    dynamic_spectra_copy[pulse_start_time:pulse_start_time + pulse.shape[0], :] += pulse_resized

    return dynamic_spectra_copy

def inject_scattered_pulse_into_dynamic_spectrum(dynamic_spectra, pulse, pulse_start_time=None):
    """
    Inject a pulse into a dynamic spectrum.

    Parameters:
    - dynamic_spectra (numpy.ndarray): The dynamic spectrum into which the pulse will be injected.
    - pulse (numpy.ndarray): The pulse to inject into the dynamic spectrum.
    - pulse_start_time (int, optional): The time sample where you want to inject the pulse.
      If None (default), it will be placed in the middle of the dynamic spectrum.

    Returns:
    - numpy.ndarray: The dynamic spectrum with the injected pulse.
    """

    # Make a copy of the dynamic spectrum
    dynamic_spectra_copy = dynamic_spectra.copy()

    # Calculate the time sample where you want to inject the pulse
    if pulse_start_time is None:
        pulse_start_time = dynamic_spectra_copy.shape[0] // 3

    # Ensure the dimensions of the pulse match the target region
    desired_shape = dynamic_spectra_copy[pulse_start_time:pulse_start_time + pulse.shape[0], :].shape
    pulse_resized = np.resize(pulse, desired_shape)

    # Inject the resized pulse into the copied dynamic spectrum
    dynamic_spectra_copy[pulse_start_time:pulse_start_time + pulse.shape[0], :] += pulse_resized

    return dynamic_spectra_copy

def inject_complex_pulse_into_dynamic_spectrum(dynamic_spectra, pulse, pulse_start_time=None):
    """
    Inject a pulse into a dynamic spectrum.

    Parameters:
    - dynamic_spectra (numpy.ndarray): The dynamic spectrum into which the pulse will be injected.
    - pulse (numpy.ndarray): The pulse to inject into the dynamic spectrum.
    - pulse_start_time (int, optional): The time sample where you want to inject the pulse.
      If None (default), it will be placed in the middle of the dynamic spectrum.

    Returns:
    - numpy.ndarray: The dynamic spectrum with the injected pulse.
    """

    # Make a copy of the dynamic spectrum
    dynamic_spectra_copy = dynamic_spectra.copy()

    # Calculate the time sample where you want to inject the pulse
    if pulse_start_time is None:
        pulse_start_time = dynamic_spectra_copy.shape[0] // 3

    # Ensure the dimensions of the pulse match the target region
    desired_shape = dynamic_spectra_copy[pulse_start_time:pulse_start_time + pulse.shape[0], :].shape
    pulse_resized = np.resize(pulse, desired_shape)

    # Inject the resized pulse into the copied dynamic spectrum
    dynamic_spectra_copy[pulse_start_time:pulse_start_time + pulse.shape[0], :] += pulse_resized

    return dynamic_spectra_copy

    
def get_dynamic_spectra_from_filterbank(file_name, num_time_samples):
    """
    Retrieve dynamic spectra from a filterbank file.

    Parameters:
        file_name (str): The name of the filterbank file to read.
        num_time_samples (int): The number of time samples to retrieve.

    Returns:
        tuple: A tuple containing the dynamic spectra data as a numpy.ndarray
        and the Your object representing the filterbank data.
    """
    temp_dir = tempfile.TemporaryDirectory()
    yr_obj = Your(file_name)
    dynamic_spectra = yr_obj.get_data(0, num_time_samples)
    return dynamic_spectra, yr_obj

def get_scaling_factor(min_value, max_value, exponent):
    """
    Generate a random scaling factor from a power law-like distribution.

    Parameters:
    - min_value (float): The minimum value of the range.
    - max_value (float): The maximum value of the range.
    - exponent (float): The exponent for the power law distribution (use a negative exponent).

    Returns:
    - float: A random scaling factor sampled from the power law-like distribution.
    """

    # Generate a random number from a power law-like distribution
    scaling_factor = ((max_value**(exponent+1) - min_value**(exponent+1)) * np.random.random(1) + min_value**(exponent+1))**(1/(exponent+1))
    
    return scaling_factor[0]  # Convert to scalar and return

def normalize(spectra):
    """
    Normalize a spectra to have values between 0 and 1.

    Parameters:
        spectra (numpy.ndarray): The input spectra.

    Returns:
        numpy.ndarray: The normalized spectra.
    """
    min_value = np.min(spectra)
    max_value = np.max(spectra)
    range_value = max_value - min_value


    normalized = (spectra - min_value) / range_value

    return normalized

def generate_scattering_time(frequency, tau_0, freq_0=1000):
    """
    Generate a scattering time that increases with decreasing frequency.
    
    Parameters:
    - frequency: The frequency at which to calculate the scattering time.
    - tau_0: The scattering time at the reference frequency freq_0.
    - freq_0: The reference frequency (default is 600 MHz).
    
    Returns:
    - tau: The scattering time at the given frequency.
    """
    return tau_0 * (frequency / freq_0) ** -4

def apply_scattering(pulse, frequencies, tau_0, time_step):
    """
    Apply scattering to the pulse by convolving each frequency channel with an exponential decay.
    
    Parameters:
    - pulse: The input pulse (time samples x frequency channels).
    - frequencies: The array of frequencies corresponding to the frequency channels.
    - tau_0: The scattering time at the reference frequency.
    - time_step: The time step size in seconds.
    
    Returns:
    - scattered_pulse: The pulse after applying scattering.
    """
    num_time_samples, num_frequency_channels = pulse.shape
    scattered_pulse = np.zeros_like(pulse)
    
    for i in range(num_frequency_channels):
        frequency = frequencies[i]
        tau = generate_scattering_time(frequency, tau_0)
        decay_function = np.exp(-np.arange(0, num_time_samples * time_step, time_step) / tau)
        decay_function /= decay_function.sum()  # Normalize to maintain total energy
        
        # Convolve the pulse with the decay function
        convolved = convolve(pulse[:, i], decay_function, mode='full')[:num_time_samples]
        
        # Ensure the convolved signal has the correct length
        if len(convolved) > num_time_samples:
            convolved = convolved[:num_time_samples]
        elif len(convolved) < num_time_samples:
            convolved = np.pad(convolved, (0, num_time_samples - len(convolved)), 'constant')
        
        scattered_pulse[:, i] = convolved
    
    return scattered_pulse

if __name__ == "__main__":
    pass
