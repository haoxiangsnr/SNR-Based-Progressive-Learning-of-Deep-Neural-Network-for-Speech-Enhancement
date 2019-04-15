import torch.nn as nn

class FCBaseModel(nn.Module):
    def __int__(self, input_frames=7, n_features=257):
        super(FCBaseModel, self).__int__()
        self.main = nn.Sequential(
            nn.Linear(n_features * input_frames, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, n_features),
        )

    def forward(self, ipt):
        return self.main(ipt)


class FCLayerWithMultipleFrames(nn.Module):
    """
    构建多帧 LPS 作为输入的全连接网络层
    """

    def __init__(self, input_frames=7, n_features=257):
        """
        初始化全连接层，以多帧 LPS 作为输入
        Args:
            input_frames: 输入的 LPS 特征帧数
            n_features: LPS 特征的维度
        """
        super(FCLayerWithMultipleFrames, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_features * input_frames, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, n_features)
        )


    def forward(self, ipt):
        return self.main(ipt)


class FCLayerWithSingleFrame(nn.Module):
    def __init__(self, n_features=257):
        """
        初始化全连接层，以单帧 LPS 作为输入
        Args:
            n_features: LPS 特征的维度
        """
        super(FCLayerWithSingleFrame, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_features, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, n_features),
        )


    def forward(self, ipt):
        return self.main(ipt)
