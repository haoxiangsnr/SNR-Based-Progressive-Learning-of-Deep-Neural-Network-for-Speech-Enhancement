import numpy as np

from torch.utils.data import Dataset


class Single_SNR_Test_Dataset(Dataset):
    """
    定义用于单信噪比的训练数据集
    """

    def __init__(self, mixture_dataset, clean_dataset, limit=None, offset=0):
        super(Single_SNR_Test_Dataset, self).__init__()

        print("Loading Mixture and Clean Dataset...")
        self.mixture_dataset: dict = np.load(mixture_dataset).item()
        self.clean_dataset: dict = np.load(clean_dataset).item()
        print("Loaded.")

        assert self.mixture_dataset.keys() == self.clean_dataset.keys(), "数据集不对齐"

        self.data_titles = list(self.mixture_dataset.keys())
        self.limit = limit
        self.offset = offset

    def __len__(self):
        if self.limit:
            return self.limit
        return len(self.data_titles)

    def __getitem__(self, item):
        if self.offset:
            item = item + self.offset

        title = self.data_titles[item]
        return self.mixture_dataset[title], self.clean_dataset[title], title
