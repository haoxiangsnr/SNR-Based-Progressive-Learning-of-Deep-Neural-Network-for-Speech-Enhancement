import numpy as np
import os
from pathlib import Path

import librosa
from torch.utils.data import Dataset

class BaselineDNNTrainDataset(Dataset):
    """
    定义 Baseline 数据集
    """
    def __init__(self, dataset, limit=None, offset=0):
        super(BaselineDNNTrainDataset, self).__init__()
        # For NPY
        # dataset_path = Path(dataset, "train.npy")
        # assert dataset_path.exists(), f"数据集 {dataset} 不存在"
        #
        # print(f"Loading NPY dataset {dataset} ...")
        # self.dataset_dict = np.load(dataset_path.as_posix()).item()
        #
        # print(f"The len of full dataset is {len(self.dataset_dict)}.")
        # print(f"The limit is {limit}.")
        # print(f"The offset is {offset}.")
        pass