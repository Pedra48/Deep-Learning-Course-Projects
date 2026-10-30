import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset.gen_datasets import gen_data
from .model import Model


class Exp:
    def __init__(self, args):
        self.args = args
        '''
        requested_device = getattr(args, "device", "auto")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU.")
        '''
        # 配置设备
        self.device = "cuda"

        # 配置checkpoints路径
        self.output_dir = Path(args.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "final.pt"

        # 配置训练集 评估集 测试集 的相应的Dataloader
        self.train_dataset, self.eval_dataset, self.test_dataset = gen_data(
            args.train_num, args.eval_num, args.test_num
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True
        )
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=args.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size)

        # 配置训练相关的内容：模型 损失函数
        self.model = Model(args).to(self.device)
        self.loss_fun = nn.MSELoss()
        
        # ？
        self.history = []
        print(f"Device: {self.device}", flush=True)
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(self.device)}", flush=True)
        print(f"Model parameters: {next(self.model.parameters()).device}", flush=True)

    def train(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma
        )

        history_path = self.output_dir / "history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["epoch", "train_mse", "val_mse", "lr"]
            )
            writer.writeheader()
            for ep in range(self.args.epoch):
                self.model.train()
                total_loss = 0.0
                total_samples = 0
                learning_rate = self.optimizer.param_groups[0]["lr"]

                for x, y in self.train_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    if ep == 0 and total_samples == 0:
                        print(f"Training batch: {x.device}, shape={tuple(x.shape)}", flush=True)
                    self.optimizer.zero_grad()
                    prediction = self.model(x)
                    loss = self.loss_fun(prediction, y)
                    loss.backward()
                    self.optimizer.step()

                    batch_size = x.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

                train_loss = total_loss / total_samples
                val_loss = self.eval()
                row = {
                    "epoch": ep + 1,
                    "train_mse": train_loss,
                    "val_mse": val_loss,
                    "lr": learning_rate,
                }
                self.history.append(row)
                writer.writerow(row)
                stream.flush()
                self.lr_scheduler.step()

                if (ep + 1) % self.args.save_epoch == 0:
                    torch.save(
                        self.model.state_dict(),
                        self.checkpoint_dir / f"epoch_{ep + 1}.pt",
                    )
                if ep == 0 or (ep + 1) % self.args.log_every == 0 or ep + 1 == self.args.epoch:
                    print(
                        f"Epoch {ep + 1}/{self.args.epoch} | "
                        f"train MSE={train_loss:.8f} | val MSE={val_loss:.8f} | "
                        f"lr={learning_rate:.6g}",
                        flush=True,
                    )

        torch.save(self.model.state_dict(), self.checkpoint_path)
        self._save_loss_curve()

    def _evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                prediction = self.model(x)
                loss = self.loss_fun(prediction, y)
                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        return total_loss / total_samples

    def eval(self):
        return self._evaluate(self.eval_loader)

    def test(self):
        state_dict = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        test_mse = self._evaluate(self.test_loader)
        print(f"Test MSE: {test_mse:.10f}", flush=True)
        return test_mse

    def _save_loss_curve(self):
        epochs = [row["epoch"] for row in self.history]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [row["train_mse"] for row in self.history], label="Training MSE")
        ax.plot(epochs, [row["val_mse"] for row in self.history], label="Validation MSE")
        ax.set(xlabel="Epoch", ylabel="MSE", title="Feedforward network fitting sin(x)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "loss_curve.png", dpi=160)
        plt.close(fig)
