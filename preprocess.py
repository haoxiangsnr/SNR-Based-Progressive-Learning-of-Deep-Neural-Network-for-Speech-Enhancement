import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

from utils.utils import add_noise_for_waveform, prepare_empty_dirs, load_wavs

def lps(y, pad=0):
    D = librosa.stft(y, n_fft=512, hop_length=256, window='hamming')
    lps = np.log(np.power(np.abs(D), 2))
    if (pad != 0):
        lps = np.concatenate((np.zeros((257, pad)), lps, np.zeros((257, pad))), axis=1)
    return lps

def corrected_length(clean_y, noise_y):
    """
    合成带噪语音前的长度矫正，使 len(clean_y) == len(noise_y)
    """
    if len(clean_y) < len(noise_y):
        return clean_y, noise_y[:len(clean_y)]
    elif len(clean_y) > len(noise_y):
        pad_factor = (len(clean_y) // len(noise_y))  # 拓展系数为需要拓展的次数，不是总数
        padded_noise_y = noise_y
        for i in range(pad_factor):
            padded_noise_y = np.concatenate((padded_noise_y, noise_y))
        noise_y = padded_noise_y
        return clean_y, noise_y[:len(clean_y)]
    else:
        return clean_y, noise_y

def main(config):
    OUTPUT_DIR = Path(config["output_dir"])
    SAMPLING_RATE = config["sampling_rate"]

    for j, dataset_cfg in enumerate(config["datasets"]):
        print(f"============ Building set {j + 1}: {dataset_cfg['name']} set ============")
        dataset_dir: Path = OUTPUT_DIR / dataset_cfg["name"]
        prepare_empty_dirs([dataset_dir])

        """============ clean speeches ============"""
        clean_meta = dataset_cfg["clean"]
        clean_speech_paths = librosa.util.find_files(
            directory=clean_meta["database"],
            ext=clean_meta["ext"],
            recurse=clean_meta["recurse"],
            limit=None,
            offset=clean_meta["offset"]
        )
        random.shuffle(clean_speech_paths)

        # 加载纯净语音时可以指定 minimum_sampling 参数，控制加载语音需要满足的最小采样点数
        # 但在加载噪声时则没有这个参数。如果在合成带噪语音阶段发现噪声长度小于语音长度，则将噪声复制多次再合成带噪语音。
        clean_ys = load_wavs(
            file_paths=clean_speech_paths,
            limit=clean_meta["limit"],
            sr=SAMPLING_RATE,
            minimum_sampling=clean_meta["minimum_sampling"],
        )
        print("Loaded clean speeches.")

        """============ noise speeches ============"""
        noise_meta = dataset_cfg["noise"]
        noise_database_dir = Path(noise_meta["database"])
        noise_ys = {}
        for noise_type in tqdm(noise_meta["types"], desc="Loading noise files"):
            noise_y, _ = librosa.load((noise_database_dir / (noise_type + ".wav")).as_posix(), sr=SAMPLING_RATE)
            noise_ys[noise_type] = noise_y

        print("Loaded noise.")

        """============ 合成 ============"""
        # 带噪
        for i, SNR in enumerate(dataset_cfg["SNRs"]):
            store = {}
            clean_store = {}
            for j, clean_y in tqdm(enumerate(clean_ys, 1), desc="Add noise for clean waveform"):
                for noise_type in noise_ys.keys():
                    output_wav_basename_text = f"{str(j).zfill(4)}_{noise_type}"
                    clean_y, noise_y = corrected_length(
                        clean_y=clean_y,
                        noise_y=noise_ys[noise_type]
                    )

                    noisy_y = add_noise_for_waveform(clean_y, noise_y, int(SNR))

                    assert len(noisy_y) == len(clean_y) == len(noise_y)

                    """
                    SNR == -5 是整个模型的输入，使用 7 帧
                    剩余的信噪比和纯净语音为模型训练的目标，使用 1 帧
                    """
                    if SNR == -5:
                        tmp_lps = torch.Tensor(lps(noisy_y, pad=3).T).unfold(0, 7, 1)
                        store[output_wav_basename_text] = tmp_lps.reshape(tmp_lps.shape[0], -1).numpy()
                    else:
                        store[output_wav_basename_text] = lps(noisy_y).T

                    if i == 0:
                        clean_store[output_wav_basename_text] = lps(clean_y).T

            print(f"Synthesize dB{SNR} finished，storing NPY file...")
            if clean_store:
                print("Saving clean NPY file...")
                np.save((dataset_dir / "clean.npy").as_posix(), clean_store)

            np.save((dataset_dir / f"dB{SNR}.npy").as_posix(), store)

if __name__ == "__main__":
    config = json.load(open("config/preprocess_config.json"))
    main(config)
