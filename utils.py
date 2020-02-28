import librosa
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def extract_mfccs(file_name, pad_len=174, n_mfcc=40):
    """Function used to extract MFCCs from audio data.

    Computes MEL Coefficients for an audio file.

    Args:
        file_name (str): Name of the audio file along with extension.
        pad_len (int): Number of samples to use.
            Default: 40 (approx. 4 seconds)
        n_mfcc (int): Number of MEL coefficients to compute.

    Returns:
        mfccs (np.array): 2D array of shape (n_mfcc, pad_len)
    """

    signal, sr = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc)

    if mfccs.shape[1] > pad_len:
        mfccs = mfccs[:, :pad_len]
    else:
        pad_width = pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')

    return mfccs


def load_data(file_name):
    """Reads the .npy file and returns X and y.

    Args:
        file_name (str): A .npy file containing data.

    Returns:
        X (np.array): Array of features.
        y (np.array): Array of labels.
    """
    data = np.load(file_name, allow_pickle=True)

    X, y = [], []

    for mfccs, label in data:
        X.append(mfccs)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(*X.shape, 1)
    y = y.reshape(-1, 1)

    return X, y


def save_history_plot(history, base_dir):
    """Saves a plot of the training history.

    Args:
        history: History object returned from model.fit
        base_dir (str): Path to directory where the plot will be saved.

    Example:
        history = model.fit(x_train, y_train)
        save_history_plot(history, 'Model')
    """

    plt.figure(figsize=(10, 10))

    plt.subplot(211)
    plt.plot(history.history['sparse_categorical_accuracy'], color='b', label='Acc')
    plt.plot(history.history['val_sparse_categorical_accuracy'], '--', color='r', label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(212)
    plt.plot(history.history['loss'], color='b', label='Loss')
    plt.plot(history.history['val_loss'], '--', color='r', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'{base_dir}\\history.png')
