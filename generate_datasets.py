import os
import numpy as np
import matplotlib.pyplot as plt
from simulate_frbs import *
import csv

def generate_simple_burst_dataset(
    save_dir,
    num_pulses,
    num_time_samples,
    exponent,
    filterbank_file,
    time_sigma_mean,
    time_sigma_std,
    freq_sigma_mean,
    freq_sigma_std
    ):
    """
    Generate a dataset of simulated simple gaussian bursts and save them as numpy arrays.

    Parameters:
        save_dir (str): The directory to save the numpy arrays.
        num_pulses (int): The number of bursts to generate.
        num_time_samples (int): The number of time samples in the filterbank file.
        exponent (float): The exponent for the power-law distribution used to scale SNR.
        filterbank_file (str): The path to the filterbank file to extract parameters from.
        time_sigma_mean (float): The mean value for pulse width in time.
        time_sigma_std (float): The standard deviation for pulse width in time.
        freq_sigma_mean (float): The mean value for frequency width.
        freq_sigma_std (float): The standard deviation for frequency width.

    Returns:
        None
    """

    # Load filterbank file to get relevant parameters
    dynamic_spectra, yr_obj = get_dynamic_spectra_from_filterbank(filterbank_file, num_time_samples=num_time_samples)

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a list to store parameters for each burst
    burst_parameters_list = []

    for i in range(num_pulses):
        # Draw random values for signal width and temporal width from Gaussian distributions
        sigma_time = np.random.normal(time_sigma_mean, time_sigma_std)
        sigma_freq = np.random.normal(freq_sigma_mean, freq_sigma_std)

        # Create the pulse object with the specified parameters
        pulse_obj = create.SimpleGaussPulse(
            sigma_time=sigma_time,
            sigma_freq=sigma_freq,
            center_freq=yr_obj.your_header.center_freq,
            dm=0,
            tau=0,
            phi=np.pi / 3,
            spectral_index_alpha=0,
            chan_freqs=yr_obj.chan_freqs,
            tsamp=yr_obj.your_header.tsamp,
            nscint=0,
            bandpass=None,
        )

        # Scale the SNR of the pulse according to a power-law distribution
        scaling_factor = get_scaling_factor(min_value=0.0002, max_value=0.003, exponent=exponent)

        # Generate the pulse signal with the specified parameters
        pulse = pulse_obj.sample_pulse(nsamp=int(3e5), dtype=np.float32)

        pulse = pulse * scaling_factor

        # Inject the pulse into the dynamic spectra
        dynamic_spectra_w_pulse = inject_pulse_into_dynamic_spectrum(dynamic_spectra, pulse)

        # Create a dictionary to store the parameters
        parameters = {
            'sigma_time': sigma_time,
            'sigma_freq': sigma_freq,
            'tau': 0,
            'exponent': exponent,
            'scaling_factor': scaling_factor,
            'num_time_samples': num_time_samples
        }

        # Append parameters for this burst to the list
        burst_parameters_list.append(parameters)

        # Define a filename for the numpy array
        filename = os.path.join(
            save_dir, f"frb_{i}.npy")

        # Save the dynamic spectra as a numpy array
        np.save(filename, dynamic_spectra_w_pulse)

    # Define a filename for the CSV file for the entire dataset
    csv_filename = os.path.join(save_dir, f"{save_dir}.csv")

    # Write all the parameters for the entire burst dataset to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = burst_parameters_list[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header row
        writer.writeheader()

        # Write parameter values for each burst
        for parameters in burst_parameters_list:
            writer.writerow(parameters)



def generate_scattered_burst_dataset(
    save_dir,
    num_pulses,
    num_time_samples,
    exponent,
    filterbank_file,
    time_sigma_mean,
    time_sigma_std,
    freq_sigma_mean,
    freq_sigma_std,
    tau_mean
    ):
    """
    Generate a dataset of simulated bursts and save them as numpy arrays.

    Parameters:
        save_dir (str): The directory to save the numpy arrays.
        num_pulses (int): The number of bursts to generate.
        num_time_samples (int): The number of time samples in the filterbank file.
        exponent (float): The exponent for the power-law distribution used to scale SNR.
        filterbank_file (str): The path to the filterbank file to extract parameters from.
        time_sigma_mean (float): The mean value for pulse width in time.
        time_sigma_std (float): The standard deviation for pulse width in time.
        freq_sigma_mean (float): The mean value for frequency width.
        freq_sigma_std (float): The standard deviation for frequency width.
        tau_mean (float): The mean scattering time

    Returns:
        None
    """

    # Load filterbank file to get relevant parameters
    dynamic_spectra, yr_obj = get_dynamic_spectra_from_filterbank(filterbank_file, num_time_samples=num_time_samples)

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a list to store parameters for each burst
    burst_parameters_list = []

    for i in range(num_pulses):
        # Draw random values for signal width and temporal width from Gaussian distributions
        sigma_time = np.random.normal(time_sigma_mean, time_sigma_std)
        sigma_freq = np.random.normal(freq_sigma_mean, freq_sigma_std)

        # Draw random value for scattering time from lognormal distribution
        max_tau = 67  # The maximum allowed value for tau
        while True:
            # Draw random value for scattering time from lognormal distribution
            tau = np.random.lognormal(tau_mean, 1, 1)
            
            if tau <= max_tau:
                break

        # Create the pulse object with the specified parameters
        pulse_obj = create.SimpleGaussPulse(
            sigma_time=sigma_time,
            sigma_freq=sigma_freq,
            center_freq=yr_obj.your_header.center_freq,
            dm=0,
            tau=tau,
            phi=np.pi / 3,
            spectral_index_alpha=0,
            chan_freqs=yr_obj.chan_freqs,
            tsamp=yr_obj.your_header.tsamp,
            nscint=0,
            bandpass=None,
        )

        # Scale the SNR of the pulse according to a power-law distribution
        scaling_factor = get_scaling_factor(min_value=0.0002, max_value=0.003, exponent=exponent)

        # Generate the pulse signal with the specified parameters
        pulse = pulse_obj.sample_pulse(nsamp=int(3e5), dtype=np.float32)

        pulse = pulse * scaling_factor

        # Inject the pulse into the dynamic spectra
        dynamic_spectra_w_pulse = inject_pulse_into_dynamic_spectrum(dynamic_spectra, pulse)

        # Create a dictionary to store the parameters
        parameters = {
            'sigma_time': sigma_time,
            'sigma_freq': sigma_freq,
            'tau': np.float(tau),
            'exponent': exponent,
            'scaling_factor': scaling_factor,
            'num_time_samples': num_time_samples
        }

        # Append parameters for this burst to the list
        burst_parameters_list.append(parameters)

        # Define a filename for the numpy array
        filename = os.path.join(
            save_dir, f"frb_{i}.npy")

        # Save the dynamic spectra as a numpy array
        np.save(filename, dynamic_spectra_w_pulse)

    # Define a filename for the CSV file for the entire dataset
    csv_filename = os.path.join(save_dir, f"{save_dir}.csv")

    # Write all the parameters for the entire burst dataset to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = burst_parameters_list[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header row
        writer.writeheader()

        # Write parameter values for each burst
        for parameters in burst_parameters_list:
            writer.writerow(parameters)




