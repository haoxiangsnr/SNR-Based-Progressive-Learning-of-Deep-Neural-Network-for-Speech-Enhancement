import numpy as np

from torch.utils.data import Dataset


class Single_SNR_Train_Dataset(Dataset):
    """
    定义用于单信噪比的训练数据集
    """

    def __init__(self, mixture_dataset, clean_dataset, target_1_dataset, target_2_dataset):
        super(Single_SNR_Train_Dataset, self).__init__()
        print("Loading Mixture Dataset...")
        self.mixture_dataset_7_frames_wise = np.concatenate(list(np.load(mixture_dataset).item().values()))
        print(f"Loaded: {self.mixture_dataset_7_frames_wise.shape}")

        print("Loading Clean Dataset...")
        self.clean_dataset_frame_wise = np.concatenate(list(np.load(clean_dataset).item().values()))
        print(f"Loaded: {self.clean_dataset_frame_wise.shape}")

        print("Loading Target 1 Dataset...")
        self.target_1_dataset_frame_wise = np.concatenate(list(np.load(target_1_dataset).item().values()))
        print(f"Loaded: {self.target_1_dataset_frame_wise.shape}")

        print("Loading Target 2 Dataset...")
        self.target_2_dataset_frame_wise = np.concatenate(list(np.load(target_2_dataset).item().values()))
        print(f"Loaded: {self.target_2_dataset_frame_wise.shape}")

    def __len__(self):
        return self.mixture_dataset_7_frames_wise.shape[0]

    def __getitem__(self, item):
        # print(item)
        return (
            self.mixture_dataset_7_frames_wise[item],
            self.target_1_dataset_frame_wise[item],
            self.target_2_dataset_frame_wise[item],
            self.clean_dataset_frame_wise[item]
        )
