import os
import numpy as np
import pandas as pd
import torch

from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler


def z_norm(signal):
    """ Normalizes the input signal of a single patient; Fits all values between 0 and 1. """
    return (signal - min(signal)) / (max(signal) - min(signal))


def butter_bandpass_filter(sig, low_cut, high_cut, fs, order=3):
    b, a = butter(order, [low_cut, high_cut], fs=fs, btype='bandpass', output='ba')
    y = filtfilt(b, a, sig)
    return y


def get_patientData(dataPath, patient, norm=True, movingAverage=False, window=10, bandpassFilter=False, low_cut=0.4,
                    high_cut=30, fs=360):
    # Load current patient signal
    signal = pd.read_csv(os.path.join(dataPath, patient + ".csv"), usecols=[0, 1])
    # Load current patient signal annotations
    annotations = pd.read_fwf(os.path.join(dataPath, patient + "annotations.txt"), usecols=[0, 1, 2])

    signal.columns = ['sample_num', 'beat']
    annotations.columns = ['time', 'sample_num', 'type']

    if norm:
        # Normalize signal to the range [0, 1]
        signal.beat = z_norm(signal.beat)

    if movingAverage:
        # Apply moving average to the signal
        signal.beat = signal.beat.rolling(int(window)).mean()

    if bandpassFilter:
        # Pass the signal through a frequency bandpass filter
        signal.beat = butter_bandpass_filter(signal.beat, low_cut, high_cut, fs, order=3)

    return signal, annotations


def split_to_windows(beat, annotations, window_size=360):
    """ This function segments the given ecg signal of a single patient to 1 second windows. It also converts the
    annotations to binary labels (0:normal window, 1:window with abnormality) and discards unclassified samples('Q',
    '?' and 'non-beat' annotations).

        Parameters ---------- beat : ecg signal of patient annotations : array-like 0bject containing the label of
        each sample of the signal window_size : size of desired window in terms of samples taken per second (default
        is 360 as the sampling rate of the dataset is 360 Hz)
    """

    # Create a dataframe where each row is one window. The last window is padded.
    windows = beat.reindex(range((beat.size // window_size + 1) * window_size), method='pad')
    windows = windows.values.reshape((-1, window_size))
    # Set normal labels
    normal_beats = ['N']
    # Set abnormality labels
    abnormal_beats = ['L', 'R', 'V', 'A', 'a', 'j', 'S', 'F', 'E', 'e', 'r', '!', 'f', '/']
    labels = [-1] * windows.size
    for index, row in annotations.iterrows():
        idx = row['sample_num']
        ann = row['type']
        if ann in normal_beats:
            labels[idx] = 0
        elif ann in abnormal_beats:
            labels[idx] = 1
    labels = np.asarray(labels)
    labels = labels.reshape((-1, window_size))
    w_labels = np.amax(labels, axis=1)
    # Find all indexes with labels <> -1
    kept_idx = np.asarray(w_labels != -1).nonzero()[0]
    # Keep relevant windows and labels (i.e., remove samples with label -1)
    k_windows = windows[kept_idx, :]
    k_labels = w_labels[kept_idx]

    return k_windows, k_labels


def resample_beats(beats, sample_size):
    """ Resamples input beats to fixed dimension. """

    beats_resampled = np.zeros((len(beats), sample_size))
    for i in range(beats_resampled.shape[0] - 1):
        beats_resampled[i] = resample(beats[i], sample_size)

    return beats_resampled


def exclude_patients(labels):
    labels_distribution = np.array(np.unique(labels, return_counts=True)).T

    if labels_distribution.shape[0] == 1:
        return True
    elif labels_distribution[1][1] / (labels_distribution[0][1] + labels_distribution[1][1]) < 0.009:
        return True
    else:
        return False


def resize_input(train_data, test_data):
    # Convert (N, D) to (N, 1, D) to fit the 1D CNN
    train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
    test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

    return train_data, test_data


def createData(opt, data_list):
    train_data, test_data, train_labels, test_labels = [], [], [], []

    for patient in data_list:
        # Get the processed signal of the current patient and its annotations
        signal, signal_notes = get_patientData(opt.data_path, patient, norm=True, movingAverage=False,
                                               bandpassFilter=True)
        # Split signal into 1 second windows
        windows, labels = split_to_windows(signal.beat, signal_notes)
        # Resample every window to the same fixed size
        windows_resampled = resample_beats(windows, opt.input_size)
        # Exclude patients belonging in extreme cases
        if exclude_patients(labels):
            continue
        # Split data to train and test sets
        train_set, test_set, train_annotations, test_annotations = train_test_split(windows_resampled, labels,
                                                                                    test_size=0.2, random_state=42, stratify=labels)
        # Fill arrays
        train_data.append(train_set)
        test_data.append(test_set)
        train_labels.extend(train_annotations)
        test_labels.extend(test_annotations)

    # Create global training and test datasets with all patients
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Resize data to fit the CNN input
    train_data, test_data = resize_input(train_data, test_data)
    # Split training data to train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

    return x_train, x_val, y_train, y_val, test_data, test_labels


def get_balanced_sampler(annotations):

    labels_distribution = np.array(np.unique(annotations, return_counts=True)).T
    # Here we create a dictionary with the class as the key and the indices on the annotations (and beats) list (array)
    # as the value
    class_counts = {row[0]: row[1] for row in labels_distribution}
    # The method we will use to balance the dataset is oversampling
    sample_weights = [1 / class_counts[i] for i in annotations]
    # Weights: kind of probability of sample to be selected
    # Num_samples: size of resulting dataset
    # Replacement = True necessary for any oversampling done
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(annotations), replacement=True)
    return sampler


def get_dataloader(data, labels, batch_size=32, drop_last=False, weightedSampling=False):
    # Set weighted sampler
    sampler = None

    if weightedSampling:
        sampler = get_balanced_sampler(labels)

    # Convert np.array to Tensor
    data = torch.tensor(data).float()
    labels = torch.tensor(labels).float()
    dataset = list(zip(list(data), list(labels)))

    if sampler is None:
        sampler = RandomSampler(dataset)
    # Get dataloader
    dataloader = DataLoader(dataset=dataset,
                            num_workers=0,
                            batch_size=batch_size,
                            pin_memory=True,
                            sampler=sampler,
                            drop_last=drop_last)
    return dataloader
