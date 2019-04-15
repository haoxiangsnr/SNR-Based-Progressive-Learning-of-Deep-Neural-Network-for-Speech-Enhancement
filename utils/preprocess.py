import numpy as np
import librosa
import os
import torch
import librosa.display
from tqdm import tqdm

def lps(y, pad=0):
    D = librosa.stft(y, n_fft=512, hop_length=256, window='hamming')
    lps = np.log(np.power(np.abs(D), 2))
    if (pad != 0):
        lps = np.concatenate((np.zeros((257, pad)), lps, np.zeros((257, pad))), axis=1)
    return lps


for set_name in ["train_set"]:
    print(f"Set Name: {set_name}")
    dataset = {"clean": {}}

    if set_name == "train_set":
        print("Loading training data...")
        data = np.load("/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/timit_single_snr/train.npy")
        data = data.item().items()
    else:
        print("Loading test data...")
        data = np.load("/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/timit_single_snr/test.npy")
        data = data.item().items()

    for i, (k, v) in tqdm(enumerate(data), desc=f"{set_name} loading"):
        # {"0001_babble_0": {"noisy": [], "clean": []}, ...}
        # k is 0001_babble_0
        num, noise, snr = k.split("_")

        if not snr in dataset:
            # {"0": {}}
            dataset[snr] = {}

        new_k = "_".join((num, noise))  # new_k is 0001_babble

        tmp_lps = torch.Tensor(lps(v["noisy"], pad=3).T).unfold(0, 7, 1)
        dataset[snr][new_k] = tmp_lps.reshape(tmp_lps.shape[0], -1).numpy()

        # clean speech
        if new_k in dataset["clean"]:
            if i % 100 == 0:
                # save time
                assert dataset["clean"][new_k].all() == lps(v["clean"]).T.all()
        else:
            dataset["clean"][new_k] = lps(v["clean"]).T

    print(dataset.keys())
    for i, (k, v) in tqdm(enumerate(dataset.items()), desc=f"{set_name} saving"):
        np.save(f"/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/timit_single_snr/{set_name}_{k}", v)
