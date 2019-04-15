import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from utils.metrics import compute_STOI, compute_PESQ
from utils.utils import ExecutionTime


plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            train_dl,
            loss_func,
            net_1,
            net_2,
            net_3,
            optim_1,
            optim_2,
            optim_3,
    ):
        super(Trainer, self).__init__(
            config,
            resume,
            loss_func,
            net_1,
            net_2,
            net_3,
            optim_1,
            optim_2,
            optim_3
        )
        self.train_data_loader = train_dl
        self.validation_data_loader = None
        self.test_data_loader = None


    def _set_model_train(self):
        # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/13
        self.net_1.train()
        self.net_2.train()
        self.net_3.train()


    def _set_model_eval(self):
        self.net_1.eval()
        self.net_2.eval()
        self.net_3.eval()


    def _train_epoch(self, epoch):
        """
        定义单次训练的逻辑
        """
        self._set_model_train()
        loss_1_total = 0.0
        loss_2_total = 0.0
        loss_3_total = 0.0
        for i, (mixture, target_1, target_2, clean) in enumerate(self.train_data_loader):
            mixture = mixture.to(self.dev)
            target_1 = target_1.to(self.dev)
            target_2 = target_2.to(self.dev)
            clean = clean.to(self.dev)

            self.optimizer_1.zero_grad()
            self.optimizer_2.zero_grad()
            self.optimizer_3.zero_grad()

            """============ Mixture  => Target 1 ============"""
            net_1_out = self.net_1(mixture)
            loss_1 = self.loss_func(net_1_out, target_1)
            loss_1.backward(retain_graph=True)
            loss_1_total += loss_1

            for p in self.net_1.parameters():
                p.grad *= 0.1

            self.optimizer_1.step()
            # 不确定更新参数后，梯度是否自动清零，手动保证一下
            self.optimizer_1.zero_grad()

            """============ Target_1 => Target 2 ============"""
            net_2_out = self.net_2(net_1_out)
            loss_2 = self.loss_func(net_2_out, target_2)
            loss_2.backward(retain_graph=True)
            loss_2_total += loss_2

            for p in self.net_1.parameters():
                p.grad *= 0.1

            for p in self.net_2.parameters():
                p.grad *= 0.1

            self.optimizer_1.step()
            self.optimizer_2.step()

            self.optimizer_1.zero_grad()
            self.optimizer_2.zero_grad()

            """============ Target 2 =>  Clean  ============"""
            net_3_out = self.net_3(net_2_out)
            loss_3 = self.loss_func(net_3_out, clean)
            loss_3.backward()
            loss_3_total += loss_3

            self.optimizer_1.step()
            self.optimizer_2.step()
            self.optimizer_3.step()

        # https://discuss.pytorch.org/t/about-the-relation-between-batch-size-and-length-of-data-loader/10510/4
        # The length of the loader will adapt to the batch_size
        dl_len = len(self.train_data_loader)
        visualize_loss = lambda tag, total: self.viz.writer.add_scalar(f"损失/{tag}", total / dl_len, epoch)
        visualize_loss("Target 1 loss", loss_1_total)
        visualize_loss("Target 2 loss", loss_2_total)
        visualize_loss("Target 3 loss", loss_3_total)

    def _test_epoch(self, epoch):
        """测试轮
        测试时使用测试集，batch_size 与 num_workers 均为 1，将每次测试后的结果保存至数组，最终返回数组，后续用于可视化
        """

        self._set_model_eval()
        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []

        with torch.no_grad():
            for i, (data, target, basename_text) in enumerate(self.test_data_loader):
                data = data.to(self.dev)
                target = target.to(self.dev)
                output = self.model(data)

                self.viz.writer.add_audio(f"语音文件/{basename_text[0]}带噪语音", data, epoch, sample_rate=16000)
                self.viz.writer.add_audio(f"语音文件/{basename_text[0]}降噪语音", output, epoch, sample_rate=16000)
                self.viz.writer.add_audio(f"语音文件/{basename_text[0]}纯净语音", target, epoch, sample_rate=16000)

                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([data, output, target]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        torch.mean(y),
                        torch.std(y),
                        torch.max(y),
                        torch.min(y)
                    ))
                    librosa.display.waveplot(y.cpu().squeeze().numpy(), sr=16000, ax=ax[j])
                plt.tight_layout()

                self.viz.writer.add_figure(f"语音波形图像/{basename_text}", fig, epoch)

                ny = data.cpu().numpy().reshape(-1)
                dy = output.cpu().numpy().reshape(-1)
                cy = target.cpu().numpy().reshape(-1)

                stoi_c_n.append(compute_STOI(cy, ny, sr=16000))
                stoi_c_d.append(compute_STOI(cy, dy, sr=16000))
                pesq_c_n.append(compute_PESQ(cy, ny, sr=16000))
                pesq_c_d.append(compute_PESQ(cy, dy, sr=16000))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.viz.writer.add_scalars(f"评价指标均值/STOI", {
            "clean 与 noisy": get_metrics_ave(stoi_c_n),
            "clean 与 denoisy": get_metrics_ave(stoi_c_d)
        }, epoch)
        self.viz.writer.add_scalars(f"评价指标均值/PESQ", {
            "clean 与 noisy": get_metrics_ave(pesq_c_n),
            "clean 与 denoisy": get_metrics_ave(pesq_c_d)
        }, epoch)

        score = (get_metrics_ave(stoi_c_d) + self._transform_pesq_range(get_metrics_ave(pesq_c_d))) / 2
        return score

    def _transform_pesq_range(self, pesq_score):
        """平移 PESQ 评价指标
        将 PESQ 评价指标的范围从 -0.5 ~ 4.5 平移为 0 ~ 1
        Args:
            pesq_score: PESQ 得分
        Returns:
            0 ~ 1 范围的 PESQ 得分
        """

        return (pesq_score + 0.5) * 2 / 10


    def _is_best_score(self, score):
        """检查当前的结果是否为最佳模型"""
        if score >= self.best_score:
            self.best_score = score
            return True
        return False


    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"============ Train epoch = {epoch} ============")
            print("[0 seconds] 开始训练...")
            timer = ExecutionTime()
            self.viz.set_epoch(epoch)

            self._train_epoch(epoch)

            # if self.visualize_metrics_period != 0 and epoch % self.visualize_metrics_period == 0:
            #     # 测试一轮，并绘制波形文件
            #     print(f"[{timer.duration()} seconds] 训练结束，开始计算评价指标...")
            #     score = self._test_epoch(epoch)
            #
            #     if self._is_best_score(score):
            #         self._save_checkpoint(epoch, is_best=True)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

            print(f"[{timer.duration()} seconds] 完成当前 Epoch.")
