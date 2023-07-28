import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from mfcc import get_mfcc


class WordDataset(Dataset):
    def __init__(self, x, y, mask):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).long()
        self.mask = torch.tensor(mask)[:, :, 0]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]


def get_word_data(folder_path):
    fpaths = []
    labels = []
    spoken = []
    for f in os.listdir(folder_path):
        for w in os.listdir(f"{folder_path}/" + f):
            fpaths.append(f"{folder_path}/" + f + "/" + w)
            labels.append(f)
            if f not in spoken:
                spoken.append(f)
    print("Words spoken:", spoken)
    return fpaths, labels, spoken


def pad_and_stack(list_of_arrays):
    max_length = max(arr.shape[0] for arr in list_of_arrays)

    padded_arrays = []
    masks = []
    for arr in list_of_arrays:
        padding_length = max_length - arr.shape[0]
        padded_array = np.pad(arr, ((0, padding_length), (0, 0)), "constant")
        mask = np.pad(
            np.ones_like(arr, dtype=bool),
            ((0, padding_length), (0, 0)),
            "constant",
            constant_values=False,
        )

        padded_arrays.append(padded_array)
        masks.append(mask)

    stacked_array = np.stack(padded_arrays)
    stacked_mask = np.stack(masks)

    return stacked_array, stacked_mask


def get_path_mfcc(path, sr, window_ms=10, overlap_pct=0.25, mel_banks=20, n_mfcc=12):
    y, sr = librosa.load(path, sr=sr)
    return get_mfcc(
        y=y,
        sr=sr,
        window_ms=window_ms,
        overlap_pct=overlap_pct,
        mel_banks=mel_banks,
        n_mfcc=n_mfcc,
    )


def prepare_hmm_dataset(
    folder_path,
    sr,
    window_ms=10,
    overlap_pct=0.25,
    mel_banks=20,
    n_mfcc=12,
    test_size=0.25,
):
    # Get raw data
    fpaths, labels, spoken = get_word_data(folder_path)
    raw_data = [
        {
            "label": label,
            "mfcc": get_path_mfcc(
                path=path,
                sr=sr,
                window_ms=window_ms,
                overlap_pct=overlap_pct,
                mel_banks=mel_banks,
                n_mfcc=n_mfcc,
            ),
        }
        for path, label in zip(fpaths, labels)
    ]

    # Initialize datasets
    train_data = {}
    test_data = {}

    # Separate data for each spoken word into training and test datasets
    for word in spoken:
        mfcc_samples = [d["mfcc"] for d in raw_data if d["label"] == word]

        # Split data into train and test set
        train_samples, test_samples = train_test_split(
            mfcc_samples, test_size=test_size
        )

        # Pad and stack MFCC train samples
        stacked_train_samples, train_mask = pad_and_stack(train_samples)

        # Add to dictionaries
        train_data[word] = (stacked_train_samples, train_mask)

        # Keep test samples as a list of arrays, along with their labels
        test_data[word] = test_samples

    # Flatten the test data and labels into two separate lists
    test_samples_flat = []
    test_labels_flat = []
    for word, samples in test_data.items():
        test_samples_flat.extend(samples)
        test_labels_flat.extend([word] * len(samples))

    return train_data, (test_samples_flat, test_labels_flat)


def prepare_nn_datasets(
    folder_path,
    sr,
    window_ms=10,
    overlap_pct=0.25,
    mel_banks=20,
    n_mfcc=12,
    test_size=0.25,
):
    fpaths, labels, spoken = get_word_data(folder_path)
    raw_data = [
        {
            "label": label,
            "mfcc": get_path_mfcc(
                path=path,
                sr=sr,
                window_ms=window_ms,
                overlap_pct=overlap_pct,
                mel_banks=mel_banks,
                n_mfcc=n_mfcc,
            ),
        }
        for path, label in zip(fpaths, labels)
    ]

    label_map = {label: i for i, label in enumerate(spoken)}
    reverse_label_map = {i: label for i, label in enumerate(spoken)}

    mfcc_samples, y = [d["mfcc"] for d in raw_data], [
        label_map[d["label"]] for d in raw_data
    ]
    x, mask = pad_and_stack(mfcc_samples)

    # Split data into train and test set
    x_train, x_test, y_train, y_test, mask_train, mask_test = train_test_split(
        x, y, mask, test_size=test_size
    )

    # Create datasets for train and test set
    train_dataset = WordDataset(x_train, y_train, mask_train)
    test_dataset = WordDataset(x_test, y_test, mask_test)

    return train_dataset, test_dataset, label_map, reverse_label_map
