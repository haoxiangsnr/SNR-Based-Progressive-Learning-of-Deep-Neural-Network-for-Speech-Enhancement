import argparse
import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.single_snr_train_dataset import Single_SNR_Train_Dataset
from data.single_snr_test_dataset import Single_SNR_Test_Dataset
from models.base_model import FCLayerWithMultipleFrames, FCLayerWithSingleFrame
import models.loss as model_loss
from trainer.trainer import Trainer

torch.manual_seed(0)
np.random.seed(0)

def main(config, resume):

    train_dataset = Single_SNR_Train_Dataset(
        mixture_dataset=config["train_dataset"]["mixture_dataset"],
        clean_dataset=config["train_dataset"]["clean_dataset"],
        target_1_dataset=config["train_dataset"]["target_1_dataset"],
        target_2_dataset=config["train_dataset"]["target_2_dataset"]
    )

    train_data_args = config["train_data"]
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_data_args["batch_size"],
        num_workers=train_data_args["num_workers"],
        shuffle=train_data_args["shuffle"]
    )

    # TODO 规范化
    # 1. 计算数据均值与方差
    # 2. 通过 DataLoader 的属性，设置是否使用均值和方差的 key
    # 3. __getitem__ 中根据 key 来判断

    valid_dataset = Single_SNR_Test_Dataset(
        mixture_dataset=config["valid_dataset"]["mixture_dataset"],
        clean_dataset=config["valid_dataset"]["clean_dataset"],
        limit=config["valid_data"]["limit"],
        offset=config["valid_data"]["offset"]
    )
    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config["valid_data"]["batch_size"],
        num_workers=config["valid_data"]["num_workers"],
        shuffle=False
    )

    net_1 = FCLayerWithMultipleFrames(input_frames=7, n_features=257)
    net_2 = FCLayerWithSingleFrame()
    net_3 = FCLayerWithSingleFrame()

    optimizer_1 = torch.optim.Adam(
        params=net_1.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(0.9, 0.999)
    )
    optimizer_2 = torch.optim.Adam(
        params=net_2.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(0.9, 0.999)
    )
    optimizer_3 = torch.optim.Adam(
        params=net_3.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(0.9, 0.999)
    )

    loss_func = getattr(model_loss, config["loss_func"])

    trainer = Trainer(
        config=config,
        resume=resume,
        train_dl=train_data_loader,
        valid_dl=valid_data_loader,
        loss_func=loss_func,
        net_1=net_1,
        net_2=net_2,
        net_3=net_3,
        optim_1=optimizer_1,
        optim_2=optimizer_2,
        optim_3=optimizer_3,
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNet For Speech Enhancement')
    parser.add_argument("-C", "--config", required=True, type=str, help="训练配置文件")
    parser.add_argument('-D', '--device', default=None, type=str, help="可以使用的GPU索引，e.g. '1,2,3'")
    parser.add_argument("-R", "--resume", action="store_true", help="是否从最近的一个断点处继续训练")
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    config = json.load(open(args.config))
    config["train_config_path"] = args.config

    main(config, resume=args.resume)