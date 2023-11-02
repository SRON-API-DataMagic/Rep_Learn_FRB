import os
import numpy as np
import matplotlib.pyplot as plt
from simulate_frbs import *
from draw_from_dist import *
import csv

def generate_simple_burst_dataset(
    save_dir,
    num_pulses,
    num_time_samples,
    exponent,
    filterbank_file,
    time_sigma_mean,
    time_sigma_std,
    freq_sigma_min,
    freq_sigma_max,
    center_freq_min,
    center_freq_max
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
        freq_sigma_min (float): The minimum value for frequency width.
        freq_sigma_max (float): The maximum value for frequency width.
        center_freq_min (float): The minimum value for center frequency in MHz.
        center_freq_max (float): The maximum value for center frequency in MHz.

    Returns:
        None
    """

    # Load filterbank file to get relevant parameters
    dynamic_spectra, yr_obj = get_dynamic_spectra_from_filterbank(filterbank_file, num_time_samples=num_time_samples)
    # Crop spectrum
    dynamic_spectra = dynamic_spectra[:, 208:720]


    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a list to store parameters for each burst
    burst_parameters_list = []

    i = 0
    while i < num_pulses:
        try:
            sigma_time = generate_sigma_time(time_sigma_mean, time_sigma_std)
            sigma_freq = generate_sigma_freq(freq_sigma_min, freq_sigma_max)
            center_freq = generate_center_freq(center_freq_min, center_freq_max)

            # Create the pulse object with the specified parameters
            pulse_obj = create.SimpleGaussPulse(
                dm=0,
                sigma_time=sigma_time,
                sigma_freq=sigma_freq,
                center_freq=center_freq,
                tau=0,
                phi=np.pi / 3,
                spectral_index_alpha=0,
                chan_freqs=yr_obj.chan_freqs,
                tsamp=yr_obj.your_header.tsamp,
                nscint=0,
                bandpass=None,
            )

            # Scale the SNR of the pulse according to a power-law distribution
            scaling_factor = get_scaling_factor(min_value=0.02, max_value=0.4, exponent=exponent)

            # Generate the pulse signal with the specified parameters
            pulse = pulse_obj.sample_pulse(nsamp=int(3e5), dtype=np.float32)

            # Crop pulse
            pulse = pulse[:, 208:720]

            # Normalize before scaling for consistency
            pulse = pulse / np.max(pulse)

            # Scale the pulse
            pulse = (pulse * scaling_factor).astype(np.float32)

            # Inject the pulse into the dynamic spectra
            dynamic_spectra_w_pulse = inject_pulse_into_dynamic_spectrum(dynamic_spectra, pulse)

            # Create a dictionary to store the parameters
            parameters = {
                'sigma_time': sigma_time,
                'sigma_freq': sigma_freq,
                'center_freq': center_freq,
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

            i += 1
        
        except ValueError:
            # Handle the error or simply continue to the next iteration
            continue

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
    freq_sigma_min,
    freq_sigma_max,
    center_freq_min,
    center_freq_max,
    tau_mean,
    tau_max
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
        freq_sigma_min (float): The minimum value for frequency width.
        freq_sigma_max (float): The maximum value for frequency width.
        center_freq_min (float): The minimum value for center frequency in MHz.
        center_freq_max (float): The maximum value for center frequency in MHz.
        tau_mean (float): The mean scattering time in time bins
        tau_max (float): The maximum scattering time in time bins

    Returns:
        None
    """

    # Load filterbank file to get relevant parameters
    dynamic_spectra, yr_obj = get_dynamic_spectra_from_filterbank(filterbank_file, num_time_samples=num_time_samples)
    # Crop spectrum
    dynamic_spectra = dynamic_spectra[:, 208:720]


    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a list to store parameters for each burst
    burst_parameters_list = []

    i = 0
    while i < num_pulses:
        try:
            sigma_time = generate_sigma_time(time_sigma_mean, time_sigma_std)
            sigma_freq = generate_sigma_freq(freq_sigma_min, freq_sigma_max)
            center_freq = generate_center_freq(center_freq_min, center_freq_max)
            scattering_time = generate_scattering_time(tau_mean, tau_max)

            # Create the pulse object with the specified parameters
            pulse_obj = create.SimpleGaussPulse(
                dm=0,
                sigma_time=sigma_time,
                sigma_freq=sigma_freq,
                center_freq=center_freq,
                tau=scattering_time,
                phi=np.pi / 3,
                spectral_index_alpha=0,
                chan_freqs=yr_obj.chan_freqs,
                tsamp=yr_obj.your_header.tsamp,
                nscint=0,
                bandpass=None,
            )

            # Scale the SNR of the pulse according to a power-law distribution
            scaling_factor = get_scaling_factor(min_value=0.02, max_value=0.4, exponent=exponent)

            # Generate the pulse signal with the specified parameters
            pulse = pulse_obj.sample_pulse(nsamp=int(3e5), dtype=np.float32)

            # Crop pulse
            pulse = pulse[:, 208:720]

            # Normalize before scaling for consistency
            pulse = pulse / np.max(pulse)

            # Scale the pulse
            pulse = (pulse * scaling_factor).astype(np.float32)

            # Inject the pulse into the dynamic spectra
            dynamic_spectra_w_pulse = inject_scattered_pulse_into_dynamic_spectrum(dynamic_spectra, pulse)

            # Create a dictionary to store the parameters
            parameters = {
                'sigma_time': sigma_time,
                'sigma_freq': sigma_freq,
                'center_freq': center_freq,
                'tau': scattering_time * yr_obj.your_header.tsamp, # Scattering time in s by muliplying bins with tsamp
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

            i += 1
        
        except ValueError:
            # Handle the error or simply continue to the next iteration
            continue

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



