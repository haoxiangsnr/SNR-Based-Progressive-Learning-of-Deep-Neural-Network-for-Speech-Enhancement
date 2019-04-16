# SNR-Based Progressive Learning of DNN for Speech Enhancement

The implementation of the paper _[SNR-Based Progressive Learning of Deep Neural Network for Speech Enhancement](https://pdfs.semanticscholar.org/8184/c50be9a5d63aed3122962650eb19e58a7515.pdf)_.

Unofficial implementation, with some differences from the original paper.

## Todo

- [x] Data pre-processing for training, validation and test
- [x] PL DNN models
- [x] Training data
- [x] Validation data
- [x] Training logic
- [x] validation logic
- [x] visualization for validation set (waveform, audio, metrics)
- [x] Config parameters
- [ ] Integrate pre-processing scripts in the project
- [ ] Global Normalization
- [ ] Test script

### Model coverage

- [x] Single-SNR
- [ ] Multi-SNR
- [ ] Baseline DNN model

## Dependencies

- tqdm
- pypesq
- pystoi
- librosa
- pytorch==1.0
- matplotlib
- tensorboardX

## Usage

#### Data Pre-processing

:sweat_smile: waiting for integration into the project.

#### Training

```bash
# specify training configuration (-C), and GPU devices (-D).
python train.py -C config/train/<name>.json -D 0

# Resume expriment (-R)
python train.py -C config/train/<name>.json -D 0 -R
```

Configuration for Training: 

```
{
    # 实验名，每次实验都不能重复
    "name": "basic_1000", 
    # 实验使用的 GPU 数量，需要配合 train.py 的 -D 选项
    "n_gpu": 1, 
    # 是否使用 Cudnn 加快速度，使用时无法保证实验的可重复性
    "use_cudnn": true,
    # 损失函数，见 models/loss.py
    "loss_func": "mse_loss",
    # Checkpoints，logs 等目录所在的位置，可以指定在另外的数据盘中
    "save_location": "/media/imucs/DataDisk/haoxiang/Experiment/DNN",
    # 本次实验的描述信息，后续会打印在 tensorboardX 中
    "description": "修复 epoch 显示错误的问题",
    # 可视化评价指标的频率
    "visualize_metrics_period": 10,
    # 优化器参数
    "optimizer": {
        "lr": 0.01
    },
    # 训练过程的可配置参数
    "trainer": {
        "epochs": 1000,
        # 模型存储的频率
        "save_period": 3
    },
    # 所有关于训练的数据集
    "train_dataset": {
        "mixture_dataset": "/home/imucs/Center/timit_single_snr_limit_500/train/dB-5.npy",
        "clean_dataset": "/home/imucs/Center/timit_single_snr_limit_500/train/clean.npy",
        "target_1_dataset": "/home/imucs/Center/timit_single_snr_limit_500/train/dB5.npy",
        "target_2_dataset": "/home/imucs/Center/timit_single_snr_limit_500/train/dB15.npy"
    },
    # 所有关于验证的数据集
    "valid_dataset": {
        "mixture_dataset": "/home/imucs/Center/timit_waveform_192/test/dB-5.npy",
        "clean_dataset": "/home/imucs/Center/timit_waveform_192/test/clean.npy"
    },
    # 训练数据集中的参数
    "train_data": {
        "batch_size": 20000,
        "shuffle": true,
        "num_workers": 100
    },
    # 验证数据集中的参数
    "valid_data": {
        "limit": 20,
        "offset": 1,
        "batch_size": 1,
        "num_workers": 1
    }
}
```

#### Visualization

```bash
tensorborad --logdir <training_config:save_location>/logs --port <port>
```

#### Test(TODO)

Similar to the validation part in `trainer/trainer.py`.