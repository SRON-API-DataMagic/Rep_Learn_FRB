import numpy as np
from simulation import *
from draw_from_dist import *
import csv
import argparse
import os

def generate_simple_burst_dataset(
    save_dir,
    burst_name,
    num_pulses,
    num_time_samples,
    exponent,
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
        burst_name (str): The name under which the different bursts will be saved
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


    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a list to store parameters for each burst
    burst_parameters_list = []

    i = 0
    while i < num_pulses:
        try:
            # Load filterbank file to get relevant parameters
            create_filterbank_with_noise("output_with_noise.fil", std_value=np.sqrt(1.0/512), mean_value=0.0, num_frequency_channels=1000, num_time_samples=1024)
            filterbank_file = "output_with_noise.fil"
            dynamic_spectra, yr_obj = get_dynamic_spectra_from_filterbank(filterbank_file, num_time_samples=num_time_samples)
            # Crop spectrum
            dynamic_spectra = dynamic_spectra[:, 208:720]

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
            # scaling_factor = get_scaling_factor(min_value=0.02, max_value=0.4, exponent=exponent)
            scaling_factor = 1

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
                save_dir, f"{burst_name}_{i}.npy")

            # Save the dynamic spectra as a numpy array
            np.save(filename, dynamic_spectra_w_pulse)

            i += 1
        
        except ValueError:
            # Handle the error or simply continue to the next iteration
            continue

    # Define a filename for the CSV file for the entire dataset
    csv_filename = os.path.join(save_dir, "simple.csv")

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
    burst_name,
    num_pulses,
    num_time_samples,
    exponent,
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
        burst_name (str): The name under which the different bursts will be saved
        num_pulses (int): The number of bursts to generate.
        num_time_samples (int): The number of time samples in the filterbank file.
        exponent (float): The exponent for the power-law distribution used to scale SNR.
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

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a list to store parameters for each burst
    burst_parameters_list = []

    i = 0
    while i < num_pulses:
        try:
            # Load filterbank file to get relevant parameters
            create_filterbank_with_noise("output_with_noise.fil", std_value=np.sqrt(1.0/512), mean_value=0.0, num_frequency_channels=1000, num_time_samples=1024)
            filterbank_file = "output_with_noise.fil"
            dynamic_spectra, yr_obj = get_dynamic_spectra_from_filterbank(filterbank_file, num_time_samples=num_time_samples)
            # Crop spectrum
            dynamic_spectra = dynamic_spectra[:, 208:720]

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
            # scaling_factor = get_scaling_factor(min_value=0.02, max_value=0.4, exponent=exponent)
            scaling_factor = 1

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
                save_dir, f"{burst_name}_{i}.npy")

            # Save the dynamic spectra as a numpy array
            np.save(filename, dynamic_spectra_w_pulse)

            i += 1
        
        except (ValueError, AssertionError) as e:
            continue

    # Define a filename for the CSV file for the entire dataset
    csv_filename = os.path.join(save_dir, "scattered_parameters.csv")

    # Write all the parameters for the entire burst dataset to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = burst_parameters_list[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header row
        writer.writeheader()

        # Write parameter values for each burst
        for parameters in burst_parameters_list:
            writer.writerow(parameters)



def generate_complex_burst_dataset(
    save_dir,
    burst_name,
    num_pulses,
    num_time_samples,
    exponent,
    time_sigma_mean,
    time_sigma_std,
    freq_sigma_min,
    freq_sigma_max,
    center_freq_min,
    center_freq_max,
    ):
    """
    Generate a dataset of simulated simple gaussian bursts and save them as numpy arrays.

    Parameters:
        save_dir (str): The directory to save the numpy arrays.
        burst_name (str): The name under which the different bursts will be saved
        num_pulses (int): The number of bursts to generate.
        num_time_samples (int): The number of time samples in the filterbank file.
        exponent (float): The exponent for the power-law distribution used to scale SNR.
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

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a list to store parameters for each burst
    burst_parameters_list = []

    i = 0
    while i < num_pulses:
        try:
            # Load filterbank file to get relevant parameters
            create_filterbank_with_noise("output_with_noise.fil", std_value=np.sqrt(1.0/512), mean_value=0.0, num_frequency_channels=1000, num_time_samples=1024)
            filterbank_file = "output_with_noise.fil"
            dynamic_spectra, yr_obj = get_dynamic_spectra_from_filterbank(filterbank_file, num_time_samples=num_time_samples)
            # Crop spectrum
            dynamic_spectra = dynamic_spectra[:, 208:720]
            num_components = np.random.randint(2, 7)  # Random number of components (2 to 6)

            sigma_freq = generate_sigma_freq(freq_sigma_min, freq_sigma_max)
            center_freq = generate_center_freq(center_freq_min, center_freq_max)

            # Generate random parameters for each component
            relative_intensities = [1 for _ in range(num_components)]
            sigma_times = [generate_sigma_time(time_sigma_mean, time_sigma_std) for _ in range(num_components)]
            sigma_freqs = [sigma_freq for _ in range(num_components)]
            pulse_thetas = [0 for _ in range(num_components)]
            center_freqs = [center_freq for _ in range(num_components)]
            # Initialize offsets list
            offsets = [0.0004]

            # Generate offsets based on sigma times distribution
            for j in range(1, num_components):
                offset = offsets[-1] + generate_sigma_time(0.001, 0.0004)  # Offset follows previous component
                offsets.append(offset)

            # Create the pulse object with the specified parameters
            dm_1 = 0
            pulse_obj_complex = create.GaussPulse(
                relative_intensities=relative_intensities,
                sigma_times=sigma_times,
                sigma_freqs=sigma_freqs,
                pulse_thetas=pulse_thetas,
                center_freqs=center_freqs,
                dm=dm_1,
                tau=0,
                offsets=offsets,  # all from start of window
                chan_freqs=yr_obj.chan_freqs,
                tsamp=yr_obj.your_header.tsamp,
                spectral_index_alpha=0,
                nscint=0,
                phi=np.pi / 3,
                bandpass=None,
            )

            # Scale the SNR of the pulse according to a power-law distribution
            # scaling_factor = get_scaling_factor(min_value=0.02, max_value=0.4, exponent=exponent)
            scaling_factor = 1
            # Generate the pulse signal with the specified parameters
            pulse = pulse_obj_complex.sample_pulse(nsamp=int(3e5), dtype=np.float32)

            # Crop pulse
            pulse = pulse[:, 208:720]

            # Normalize before scaling for consistency
            pulse = pulse / np.max(pulse)

            # Scale the pulse
            pulse = (pulse * scaling_factor).astype(np.float32)

            # Inject the pulse into the dynamic spectra
            dynamic_spectra_w_pulse = inject_complex_pulse_into_dynamic_spectrum(dynamic_spectra, pulse)

            # Create a dictionary to store the parameters
            parameters = {
                'sigma_time': sigma_times,
                'sigma_freq': sigma_freq,
                'center_freq': center_freq,
                'tau': 0, # Scattering time in s by muliplying bins with tsamp
                'exponent': exponent,
                'scaling_factor': scaling_factor,
                'num_time_samples': num_time_samples,
                'num_components': num_components,
                'offsets': offsets
            }

            # Append parameters for this burst to the list
            burst_parameters_list.append(parameters)

            # Define a filename for the numpy array
            filename = os.path.join(
                save_dir, f"{burst_name}_{i}.npy")
            # Save the dynamic spectra as a numpy array
            np.save(filename, dynamic_spectra_w_pulse)
            i += 1
        
        except (ValueError, AssertionError) as e:
            continue

    # Define a filename for the CSV file for the entire dataset
    csv_filename = os.path.join(save_dir, "complex_parameters.csv")

    # Write all the parameters for the entire burst dataset to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = burst_parameters_list[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header row
        writer.writeheader()

        # Write parameter values for each burst
        for parameters in burst_parameters_list:
            writer.writerow(parameters)


def generate_drifting_burst_dataset(
    save_dir,
    burst_name,
    num_pulses,
    num_time_samples,
    exponent,
    time_sigma_mean,
    time_sigma_std,
    freq_sigma_min,
    freq_sigma_max,
    center_freq_min,
    center_freq_max,
    ):
    """
    Generate a dataset of simulated simple gaussian bursts and save them as numpy arrays.

    Parameters:
        save_dir (str): The directory to save the numpy arrays.
        burst_name (str): The name under which the different bursts will be saved
        num_pulses (int): The number of bursts to generate.
        num_time_samples (int): The number of time samples in the filterbank file.
        exponent (float): The exponent for the power-law distribution used to scale SNR.
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

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a list to store parameters for each burst
    burst_parameters_list = []

    i = 0
    while i < num_pulses:
        try:
            # Load filterbank file to get relevant parameters
            create_filterbank_with_noise("output_with_noise.fil", std_value=np.sqrt(1.0/512), mean_value=0.0, num_frequency_channels=1000, num_time_samples=1024)
            filterbank_file = "output_with_noise.fil"
            dynamic_spectra, yr_obj = get_dynamic_spectra_from_filterbank(filterbank_file, num_time_samples=num_time_samples)
            # Crop spectrum
            dynamic_spectra = dynamic_spectra[:, 208:720]

            num_components = np.random.randint(4, 7)  # Random number of components (2 to 6)

            sigma_freq = generate_sigma_freq(freq_sigma_min, freq_sigma_max)
            center_freq = 0
            # Keep generating center_freq until it's above 1500
            while center_freq <= 1500:
                center_freq = generate_center_freq(1228, 1700)


            # Generate random parameters for each component
            relative_intensities = [1 for _ in range(num_components)]
            sigma_times = [generate_sigma_time(time_sigma_mean, time_sigma_std) for _ in range(num_components)]
            sigma_freqs = [sigma_freq for _ in range(num_components)]
            pulse_thetas = [0 for _ in range(num_components)]
            center_freqs = [center_freq for _ in range(num_components)]
            # Initialize offsets list
            offsets = [0.0004]

            # Generate offsets based on sigma times distribution
            for j in range(1, num_components):
                offset = offsets[-1] + generate_sigma_time(0.0003, 0.0001)  # Offset follows previous component
                offsets.append(offset)

            drift_rate = 200 # MHz/s

            # Calculate the drifts for components 2 and onward
            drifts = [0]  # Initialize drifts with zero for the first component
            for k in range(1, len(offsets)):
                time_diff_ms = (offsets[k] - offsets[0]) * 1000  # Time difference in ms relative to the first component
                drift = time_diff_ms * drift_rate
                drifts.append(drift)

            # Adjust the center frequencies for each component based on the drift
            for l in range(1, len(center_freqs)):
                center_freqs[l] -= drifts[l]  # Adjust center frequency based on drift

            # Create the pulse object with the specified parameters
            dm_1 = 0
            pulse_obj_complex = create.GaussPulse(
                relative_intensities=relative_intensities,
                sigma_times=sigma_times,
                sigma_freqs=sigma_freqs,
                pulse_thetas=pulse_thetas,
                center_freqs=center_freqs,
                dm=dm_1,
                tau=0,
                offsets=offsets,  # all from start of window
                chan_freqs=yr_obj.chan_freqs,
                tsamp=yr_obj.your_header.tsamp,
                spectral_index_alpha=0,
                nscint=0,
                phi=np.pi / 3,
                bandpass=None,
            )

            # Scale the SNR of the pulse according to a power-law distribution
            # scaling_factor = get_scaling_factor(min_value=0.02, max_value=0.4, exponent=exponent)
            scaling_factor = 1
            # Generate the pulse signal with the specified parameters
            pulse = pulse_obj_complex.sample_pulse(nsamp=int(3e5), dtype=np.float32)

            # Crop pulse
            pulse = pulse[:, 208:720]

            # Normalize before scaling for consistency
            pulse = pulse / np.max(pulse)

            # Scale the pulse
            pulse = (pulse * scaling_factor).astype(np.float32)

            # Inject the pulse into the dynamic spectra
            dynamic_spectra_w_pulse = inject_complex_pulse_into_dynamic_spectrum(dynamic_spectra, pulse)

            # Create a dictionary to store the parameters
            parameters = {
                'sigma_time': sigma_times,
                'sigma_freq': sigma_freq,
                'center_freq': center_freq,
                'tau': 0, # Scattering time in s by muliplying bins with tsamp
                'exponent': exponent,
                'scaling_factor': scaling_factor,
                'num_time_samples': num_time_samples,
                'num_components': num_components,
                'offsets': offsets
            }

            # Append parameters for this burst to the list
            burst_parameters_list.append(parameters)

            # Define a filename for the numpy array
            filename = os.path.join(
                save_dir, f"{burst_name}_{i}.npy")

            # Save the dynamic spectra as a numpy array
            np.save(filename, dynamic_spectra_w_pulse)
            i += 1
        
        except (ValueError, AssertionError) as e:
            continue

    # Define a filename for the CSV file for the entire dataset
    csv_filename = os.path.join(save_dir, "drifting_parameters.csv")

    # Write all the parameters for the entire burst dataset to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = burst_parameters_list[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header row
        writer.writeheader()

        # Write parameter values for each burst
        for parameters in burst_parameters_list:
            writer.writerow(parameters)


def generate_datasets(num_frbs, save_dir):
    """
    Generate a dataset of dynamic spectra of five different classes of 
    FRBs. Generates `num_frbs` FRB files *per class* and saves them 
    in `sav_dir`.

    Parameters
    ----------
    num_frbs : int or iterable
       The number of FRBs to simulate. If integer, simulate the 
       same number of bursts for all classes. Otherwise must be 
       an iterable of length 5.

    sav_dir : str
       The full path of the folder to store the generated FRB 
       dynamic spectra. One file per spectrum.
    """

    if np.size(num_frbs) == 1:
        num_frbs = np.zeros(5) + num_frbs

    if np.size(num_frbs) > 1 and np.size(num_frbs) < 5:
        raise ValueError("num_frbs must be integer or iterable of length 5")

    save_dir_simple_narrow = save_dir + f'/{int(num_frbs[0])}_Constant_SNR_simple_narrow/'
    save_dir_simple_broad = save_dir + f'/{int(num_frbs[1])}_Constant_SNR_simple_broad/'
    save_dir_scattered = save_dir + f'/{int(num_frbs[2])}_Constant_SNR_scattered/'
    save_dir_complex = save_dir + f'/{int(num_frbs[3])}_Constant_SNR_complex/'
    save_dir_drifting = save_dir + f'/{int(num_frbs[4])}_Constant_SNR_drifting/'

    # Call the function to generate the dataset
    generate_simple_burst_dataset(
        save_dir=save_dir_simple_narrow,
        burst_name='SN',
        num_pulses=num_frbs[0],
        num_time_samples=1024,
        exponent=-1.5,
        time_sigma_mean=0.0008,
        time_sigma_std=0.0004,
        freq_sigma_min=12.5,
        freq_sigma_max=100,
        center_freq_min=1228,
        center_freq_max=1700
    )
    
    # Call the function to generate the dataset
    generate_simple_burst_dataset(
        save_dir=save_dir_simple_broad,
        burst_name='SB',
        num_pulses=num_frbs[1],
        num_time_samples=1024,
        exponent=-1.5,
        time_sigma_mean=0.0008,
        time_sigma_std=0.0004,
        freq_sigma_min=150,
        freq_sigma_max=200,
        center_freq_min=1300,
        center_freq_max=1625
    )
    
    # Call the function to generate the dataset
    generate_scattered_burst_dataset(
        save_dir=save_dir_scattered,
        burst_name='SC',
        num_pulses=num_frbs[2],
        num_time_samples=1024,
        exponent=-1.5,
        time_sigma_mean=0.0008,
        time_sigma_std=0.0004,
        freq_sigma_min=12.5,
        freq_sigma_max=100,
        center_freq_min=1228,
        center_freq_max=1700,
        tau_mean=3,
        tau_max=100
    )
    
    # Call the function to generate the dataset
    generate_complex_burst_dataset(
        save_dir=save_dir_complex,
        burst_name='CP',
        num_pulses=num_frbs[3],
        num_time_samples=1024,
        exponent=-1.5,
        time_sigma_mean=0.0008,
        time_sigma_std=0.0004,
        freq_sigma_min=12.5,
        freq_sigma_max=100,
        center_freq_min=1228,
        center_freq_max=1700,
    )
    
    # Call the function to generate the dataset
    generate_drifting_burst_dataset(
        save_dir=save_dir_drifting,
        burst_name='DD',
        num_pulses=num_frbs[4],
        num_time_samples=1024,
        exponent=-1.5,
        time_sigma_mean=0.0008,
        time_sigma_std=0.0004,
        freq_sigma_min=12.5,
        freq_sigma_max=100,
        center_freq_min=1228,
        center_freq_max=1700,
    )

    return

def main():
    generate_datasets(num_frbs, save_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-frbs", dest="num_frbs", action="store", default=1000, type=int, required = False, help="The number of FRBs to simulate per class.")
    parser.add_argument("-s", "--save-dir", dest="save_dir", action="store", required=True, help="The directory path where to save the simulated FRBs.")

    clargs = parser.parse_args()
    num_frbs = clargs.num_frbs
    save_dir = clargs.save_dir
    main()