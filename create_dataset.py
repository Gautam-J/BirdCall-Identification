import numpy as np
import os
from tqdm import tqdm
import librosa
import librosa.display

N_MFCC = 40  # Number of MEL coefficients to compute
PAD_LEN = 174  # Number of samples to use (approx. 4 seconds)
SPLIT_NUMBER = 5  # Number of splits to perform on one file

data = []

for species in os.listdir('Birdcalls'):
    species_dir = os.path.join('Birdcalls', species)

    print('Species:', species)

    for recording in tqdm(os.listdir(species_dir)):
        file_name = os.path.join(species_dir, recording)
        signal, sr = librosa.load(file_name, res_type='kaiser_fast')

        for cut_signal in np.array_split(signal, SPLIT_NUMBER):
            mfccs = librosa.feature.mfcc(cut_signal, sr=sr, n_mfcc=N_MFCC)

            if mfccs.shape[1] > PAD_LEN:
                mfccs = mfccs[:, :PAD_LEN]
            else:
                pad_width = PAD_LEN - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')

            data.append([mfccs, species])

print('Saving data...')
np.save('data.npy', data)
print('DONE')
