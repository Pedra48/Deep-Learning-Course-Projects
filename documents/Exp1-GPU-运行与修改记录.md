# 实验一：本机 GPU 运行与修改记录

日期：2026-10-27。

本次目标是修复当前代码并使用本机显卡完整跑通一组实验。已完成一次 300 epoch 的训练、逐轮验证和一次最终测试，程序正常退出，退出码为 0。本记录中的修改以本次开始运行前的工作区内容为基准，不把此前你已经完成的改动归为本次修改。

## 1. 实际运行环境

| 项目 | 本次实际值 |
| --- | --- |
| Conda 环境 | `Deep-Learning` |
| Python | 3.12.16 |
| Python 路径 | `C:\Users\12290\miniconda3\envs\Deep-Learning\python.exe` |
| 显卡 | NVIDIA GeForce RTX 4060 Ti，8 GB 显存 |
| NVIDIA 驱动 | 595.95 |
| PyTorch | `2.8.0+cu128` |
| PyTorch CUDA runtime | 12.8 |
| NumPy | 2.5.4 |
| PyYAML | 6.0.3 |
| Matplotlib | 3.11.2 |

本次在上述 Conda 环境中安装了缺少的 PyYAML 和 Matplotlib；原有 PyTorch、NumPy 保持不变。安装命令：

```powershell
& 'C:/Users/12290/miniconda3/envs/Deep-Learning/python.exe' -m pip install 'PyYAML>=6.0,<7.0' matplotlib
```

该安装同时补充了依赖：contourpy 1.4.0、cycler 0.12.1、fonttools 4.67.0、kiwisolver 1.5.1、pillow 13.0.0、pyparsing 3.3.3、python-dateutil 3.0.0、six 1.17.0、tzdata 2026.5。

## 2. 本次实验配置

配置文件：[`configs/exp1.yaml`](../configs/exp1.yaml)。

| 参数 | 值 |
| --- | --- |
| 函数与范围 | y = sin(x)，x ∈ [0, 2π) |
| 训练 / 验证 / 测试样本数 | 1000 / 200 / 200 |
| batch_size | 64 |
| 隐藏层 | 3 层，每层 64 个神经元 |
| 激活函数 | ReLU |
| 优化器 | AdamW，使用默认 weight_decay=0.01 |
| 初始学习率 | 0.001 |
| epoch | 300 |
| StepLR | 每 100 个 epoch 将学习率乘以 0.5 |
| 随机种子 | 42 |
| 设备 | cuda，CUDA 不可用时明确报错 |
| 日志间隔 | 25 个 epoch，同时打印首轮和末轮 |
| 模型保存间隔 | 100 个 epoch，另保存 final.pt |

当前模型的 `n_layers=2` 表示中间的两个隐藏层，加上 encoder 中的一个隐藏层，合计 3 个隐藏层。此处沿用原有模型定义。

## 3. 所有代码和配置修改

### 3.1 `src/exp.py`

1. 将设备选择接到 YAML 的 `device` 参数，支持 `cuda`、`cpu` 和 `auto`；本次明确要求 CUDA，不会静默改用 CPU。
2. 训练、验证和测试中的输入、标签均使用 `.to(self.device)`，修复原来训练写死 `.cuda()`、验证和测试未移动设备的问题。
3. 训练集 DataLoader 增加 `shuffle=True`；验证和测试保持默认不打乱。
4. 将 `nn.MSELoss()` 移到初始化阶段，避免评估方法依赖于训练方法先设置损失函数。
5. 保留每个 batch 清零梯度、前向传播、反向传播、更新参数的流程；AdamW 的学习率改用显式的 `lr=` 关键字。
6. 每个 epoch 开始调用 `self.model.train()`，避免上一轮验证后模型一直处于评估模式。
7. 每个 epoch 结束进行验证，按样本数加权计算 MSE，记录训练 MSE、验证 MSE、学习率和 epoch。
8. 抽取 `_evaluate(loader)`，统一训练结束后的评估、验证和测试计算；初始化总损失和样本数，使用 `model.eval()` 和 `torch.no_grad()`，返回 Python 数值。
9. 增加 `history.csv` 的逐轮保存及控制台日志；每轮刷新 CSV，以便查看进行中的结果。
10. 统一输出和检查点路径，移除未定义的 `self.name`，修复保存目录与加载目录不一致的问题。保存第 100、200、300 轮参数及 `final.pt`。
11. 使用 `(ep + 1) % save_epoch` 判断保存轮次，修正从第 0 轮开始判断带来的编号问题。
12. 测试前加载 `final.pt`，指定 `map_location=self.device` 和 `weights_only=True`；返回并打印测试 MSE。
13. 使用 Matplotlib 的 Agg 后端保存训练/验证损失曲线，使用对数纵轴展示不同数量级的变化，并关闭图像释放资源。
14. 输出显卡名称、模型参数所在设备和首个训练 batch 的设备，留下 GPU 执行证据。
15. 移除不再使用的 `os`、`Myloss` 导入，整理类和循环的格式。

### 3.2 `main.py`

1. 保留已有的 `--config`、YAML 读取和 `SimpleNamespace` 参数传递方式。
2. 根据配置固定 Python、NumPy、PyTorch 及 CUDA 的随机种子。固定种子便于复现，但不承诺不同硬件或软件版本之间逐位相同。
3. 将相对的 `output_dir` 按项目根目录解析，创建输出目录。
4. 保存本次实际参数到输出目录中的 `config.yaml`，其中输出目录记录为解析后的绝对路径。
5. 训练完成后，额外计算最终固定模型在训练集和验证集上的 MSE。
6. 整个运行中只调用一次 `exp.test()`。没有使用测试集选择 epoch 或调整参数。
7. 保存 `metrics.json`，包含训练/验证/测试 MSE、设备、显卡、PyTorch/CUDA 版本、种子、epoch 数、运行耗时和峰值 CUDA 张量显存。
8. 结束时打印最终训练/验证 MSE 和输出目录。

### 3.3 `configs/exp1.yaml`

- `epoch` 从 500 改为 300。
- 新增 `device: cuda`、`seed: 42`、`log_every: 25`。
- 新增 `output_dir: outputs/exp1_gpu_20261027`。
- 其余已有参数保持不变。

### 3.4 `src/dataset/gen_datasets.py`

- 将 `PI = 3.14` 改为 `PI = math.pi`，使用更精确的 π。
- 沿用已有的无放回采样、Tensor 转换、`reshape(-1, 1)` 和 `TensorDataset(...)`。

### 3.5 `requirements.txt`

从只声明 PyYAML，改为记录本次代码实际使用的四个直接依赖及实际版本：

```text
torch==2.8.0
numpy==2.5.4
PyYAML==6.0.3
matplotlib==3.11.2
```

`torch==2.8.0` 是包版本要求；CUDA 12.8 构建通过下方的专用安装源选择，本机实际安装版本是 `2.8.0+cu128`。

### 3.6 `.gitignore`

- 新增 `outputs/`，将运行结果和模型参数排除在 Git 自动追踪之外。
- 输出文件仍实际保留在本机，可以直接打开。

### 3.7 新增记录文件

- 新增本文档，记录本次全部修改、环境、命令、结果和检查方式。
- `src/model.py` 和 `src/utils/loss.py` 本次没有修改；占位的 `Myloss` 已不再被实验代码导入或调用。

## 4. 实际运行命令

工作目录为 `Deep-Learning-Course-Projects`。本次通过 PowerShell 保存了完整终端日志：

```powershell
& 'C:/Users/12290/miniconda3/envs/Deep-Learning/python.exe' -B -u main.py --config configs/exp1.yaml 2>&1 | Tee-Object -FilePath 'outputs/exp1_gpu_20261027/run.log'
```

今后可在项目根目录中这样运行：

```powershell
conda activate Deep-Learning
python -u main.py --config configs/exp1.yaml
```

若在新的环境中安装依赖：

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

同一个 `output_dir` 再次运行会覆盖该目录中的同名结果文件。若要保留多次实验，请先给每次实验配置不同的输出目录。普通运行会打印日志；上面的 `Tee-Object` 命令才会额外将终端输出保存为 `run.log`。

## 5. 实际结果

| 指标 | 实测值 |
| --- | --- |
| 最终模型训练集 MSE | 2.418972027953714e-6 |
| 最终模型验证集 MSE | 2.5083617629206854e-6 |
| 测试集 MSE | 2.8407908894223512e-6 |
| 第 1 轮验证 MSE | 0.24123715（控制台显示精度） |
| 完成轮数 | 300 |
| 最终模型参数设备 | cuda:0 |
| 首个训练 batch 设备 | cuda:0 |
| 峰值 CUDA 张量显存 | 17.4561 MiB |
| 程序内计时 | 约 12.99 秒 |
| 进程退出码 | 0 |

程序内计时从创建 Exp 前开始，覆盖模型和数据初始化、训练、曲线保存、最终评估及 CUDA 同步，不包括此前的 Python 启动和模块导入；显存值是 PyTorch 分配张量的峰值，不代表显卡全部使用量。

CSV 和曲线中的训练 MSE，是该轮训练过程中各个 batch 损失按样本数加权得到的平均值；最终训练 MSE 是所有训练更新结束后，使用固定模型重新计算的结果，所以第 300 轮训练 MSE 与上表的最终训练 MSE有所不同。验证曲线使用每轮结束后的固定模型计算。

这是单组参数的运行验证，没有开展深度、宽度、学习率、激活函数的对比实验，也没有声称该配置最优。后续完成课程作业时，还需要设计验证集上的超参数对比、选定配置后重新训练，以及完成实验报告。

## 6. 输出文件

输出目录：`outputs/exp1_gpu_20261027/`。

| 文件 | 用途 |
| --- | --- |
| [config.yaml](../outputs/exp1_gpu_20261027/config.yaml) | 本次实际配置 |
| [history.csv](../outputs/exp1_gpu_20261027/history.csv) | 300 轮训练和验证记录 |
| [loss_curve.png](../outputs/exp1_gpu_20261027/loss_curve.png) | 训练和验证损失曲线 |
| [metrics.json](../outputs/exp1_gpu_20261027/metrics.json) | 最终指标和运行环境 |
| [run.log](../outputs/exp1_gpu_20261027/run.log) | 本次运行的终端输出 |
| [checkpoints/final.pt](../outputs/exp1_gpu_20261027/checkpoints/final.pt) | 最终模型参数，测试时已成功加载 |
| checkpoints/epoch_100.pt、epoch_200.pt、epoch_300.pt | 指定轮次的模型参数 |
| source_before_changes.json | 本次修改前的六个源码/配置文件快照，用于生成下方差异 |

![训练与验证损失曲线](../outputs/exp1_gpu_20261027/loss_curve.png)

## 7. 完成后的检查

- 5 个 Python 文件均通过 AST 语法检查，命令行 `--help` 可正常运行。
- 实际 GPU 训练进程正常结束；日志确认模型和训练 batch 都位于 `cuda:0`。
- CSV 包含完整的第 1 至 300 轮，训练/验证损失均为有限非负数，最后一轮验证值与最终指标一致。
- 用相同种子重新生成数据，确认样本数为 1000/200/200、Tensor 为 float32 且形状为 [N, 1]，输入落在要求区间，标签符合 sin(x)，三组输入在 float32 转换后也两两不相交。
- 本次只对测试模型执行了一次测试集评估；检查数据划分时没有再次进行模型预测。
- 配置、CSV、曲线、JSON、日志和 4 个检查点均存在且非空；曲线已打开检查，坐标、图例和曲线清晰可读。
- `git diff --check` 通过。
- 最终检查点 SHA256：`6905f62759f6cdda8c3161a457a6851b22dbd41ee82fa7aadf88cb66b9d589d3`。

## 8. 本次修改的完整差异

以下差异相对于本次运行前保存的工作区快照生成，覆盖上述六个被修改的源码/配置文件；本文档自身为新增文件。已有的 YAML 入口和 TensorDataset 修复不重复记作本次修改。


```diff
--- before/main.py
+++ after/main.py
@@ -1,5 +1,8 @@
 import argparse
+import json
+import random
 from pathlib import Path
+from time import perf_counter
 from types import SimpleNamespace
 
 
@@ -27,12 +30,56 @@
     cli_args = parser.parse_args()
     args = load_config(cli_args.config)
 
+    import numpy as np
+    import torch
+    import yaml
     from src.exp import Exp
 
+    random.seed(args.seed)
+    np.random.seed(args.seed)
+    torch.manual_seed(args.seed)
+    if torch.cuda.is_available():
+        torch.cuda.manual_seed_all(args.seed)
+
+    output_dir = Path(args.output_dir)
+    if not output_dir.is_absolute():
+        output_dir = Path(__file__).resolve().parent / output_dir
+    args.output_dir = str(output_dir)
+    output_dir.mkdir(parents=True, exist_ok=True)
+    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
+        yaml.safe_dump(vars(args), stream, sort_keys=False)
+
+    started = perf_counter()
     exp = Exp(args)
     exp.train()
-    exp.eval()
-    exp.test()
+    final_train_mse = exp._evaluate(exp.train_loader)
+    final_val_mse = exp.eval()
+    test_mse = exp.test()  # Only evaluate the test set once in this run.
+    if exp.device.type == "cuda":
+        torch.cuda.synchronize(exp.device)
+    metrics = {
+        "device": str(exp.device),
+        "model_device": str(next(exp.model.parameters()).device),
+        "gpu": torch.cuda.get_device_name(exp.device) if exp.device.type == "cuda" else None,
+        "torch_version": torch.__version__,
+        "cuda_version": torch.version.cuda,
+        "seed": args.seed,
+        "epochs": args.epoch,
+        "final_train_mse": final_train_mse,
+        "final_val_mse": final_val_mse,
+        "test_mse": test_mse,
+        "test_evaluations": 1,
+        "elapsed_seconds": perf_counter() - started,
+        "peak_cuda_memory_mb": (
+            torch.cuda.max_memory_allocated(exp.device) / 1024 ** 2
+            if exp.device.type == "cuda" else 0.0
+        ),
+    }
+    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
+        json.dump(metrics, stream, ensure_ascii=False, indent=2)
+    print(f"Final train MSE: {final_train_mse:.10f}", flush=True)
+    print(f"Final validation MSE: {final_val_mse:.10f}", flush=True)
+    print(f"Results saved to: {output_dir}", flush=True)
     
 
 if __name__ == "__main__":
--- before/configs/exp1.yaml
+++ after/configs/exp1.yaml
@@ -10,7 +10,7 @@
 
 # Training (write the learning rate as a decimal so it is parsed as a number)
 lr: 0.001
-epoch: 500
+epoch: 300
 
 # StepLR: multiply the learning rate by gamma every step_size epochs
 step_size: 100
@@ -18,3 +18,9 @@
 
 # Checkpoint interval in epochs
 save_epoch: 100
+
+# Require the local NVIDIA GPU for this run
+device: cuda
+seed: 42
+log_every: 25
+output_dir: outputs/exp1_gpu_20261027
--- before/src/exp.py
+++ after/src/exp.py
@@ -1,107 +1,150 @@
-# train eval test
+import csv
+from pathlib import Path
+
+import matplotlib
+
+matplotlib.use("Agg")
+import matplotlib.pyplot as plt
 import torch
 import torch.nn as nn
-import os
-from pathlib import Path
-
+from torch.utils.data import DataLoader
 
 from .dataset.gen_datasets import gen_data
 from .model import Model
-from .utils.loss import Myloss
 
-class Exp():
-    def __init__(self,args):
+
+class Exp:
+    def __init__(self, args):
         self.args = args
-        self.batch_size = self.args.batch_size
-        
-        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
-        # 提取 Data
-        self.train_dataset,self.eval_dataset,self.test_dataset = gen_data(args.train_num,args.eval_num,args.test_num)
-        self.train_loader = torch.utils.data.DataLoader(
-            dataset=self.train_dataset,
-            batch_size=self.batch_size,
+        requested_device = getattr(args, "device", "auto")
+        if requested_device == "auto":
+            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
+        self.device = torch.device(requested_device)
+        if self.device.type == "cuda" and not torch.cuda.is_available():
+            raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU.")
+
+        self.output_dir = Path(args.output_dir)
+        self.checkpoint_dir = self.output_dir / "checkpoints"
+        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
+        self.checkpoint_path = self.checkpoint_dir / "final.pt"
+
+        self.train_dataset, self.eval_dataset, self.test_dataset = gen_data(
+            args.train_num, args.eval_num, args.test_num
         )
-        self.eval_loader = torch.utils.data.DataLoader(
-            dataset = self.eval_dataset,
-            batch_size=self.batch_size,
+        self.train_loader = DataLoader(
+            self.train_dataset, batch_size=args.batch_size, shuffle=True
         )
-        self.test_loader = torch.utils.data.DataLoader(
-            dataset=self.test_dataset,
-            batch_size=self.batch_size
+        self.eval_loader = DataLoader(self.eval_dataset, batch_size=args.batch_size)
+        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size)
+
+        self.model = Model(args).to(self.device)
+        self.loss_fun = nn.MSELoss()
+        self.history = []
+        print(f"Device: {self.device}", flush=True)
+        if self.device.type == "cuda":
+            print(f"GPU: {torch.cuda.get_device_name(self.device)}", flush=True)
+        print(f"Model parameters: {next(self.model.parameters()).device}", flush=True)
+
+    def train(self):
+        self.optimizer = torch.optim.AdamW(
+            self.model.parameters(), lr=self.args.lr
         )
-        
-        # 初始化 Model
-        self.model = Model(args).to(self.device)
-        
-        
-    def train(self):
-        self.model.train()
-        # 1. 优化器设置
-        self.optimizer = torch.optim.AdamW(
-            self.model.parameters(),
-            self.args.lr
+        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
+            self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma
         )
-        # 2. 学习率调节器设置
-        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
-            optimizer=self.optimizer,
-            step_size= self.args.step_size,
-            gamma= self.args.gamma
-        )
-        # 3. 损失函数设置
-        # self.loss_fun = Myloss()
-        self.loss_fun = nn.MSELoss()
-        # 4. 开始训练
-        for ep in range(self.args.epoch):
-            
-            # 4.1 提取data
-            for batch in self.train_loader:
-                self.optimizer.zero_grad()
-                x,y = batch
-                x = x.cuda()
-                y = y.cuda()
-                # 4.2 forward
-                yx = self.model(x)
-                # 4.3 计算损失
-                loss = self.loss_fun(y,yx)
-                # 4.4 backward
-                loss.backward()
-                
-                # 4.5 优化器step in
-                self.optimizer.step()
-            self.lr_scheduler.step()
-            
-            if ep%self.args.save_epoch == 0:
-                if not os.path.exists('./checkpoints'):
-                    os.makedirs('./checkpoints')
-                torch.save(self.model.state_dict(),os.path.join('./checkpoints/'+self.name  + '.pt'))
-                
-    def eval(self):
+
+        history_path = self.output_dir / "history.csv"
+        with history_path.open("w", newline="", encoding="utf-8") as stream:
+            writer = csv.DictWriter(
+                stream, fieldnames=["epoch", "train_mse", "val_mse", "lr"]
+            )
+            writer.writeheader()
+            for ep in range(self.args.epoch):
+                self.model.train()
+                total_loss = 0.0
+                total_samples = 0
+                learning_rate = self.optimizer.param_groups[0]["lr"]
+
+                for x, y in self.train_loader:
+                    x = x.to(self.device)
+                    y = y.to(self.device)
+                    if ep == 0 and total_samples == 0:
+                        print(f"Training batch: {x.device}, shape={tuple(x.shape)}", flush=True)
+                    self.optimizer.zero_grad()
+                    prediction = self.model(x)
+                    loss = self.loss_fun(prediction, y)
+                    loss.backward()
+                    self.optimizer.step()
+
+                    batch_size = x.size(0)
+                    total_loss += loss.item() * batch_size
+                    total_samples += batch_size
+
+                train_loss = total_loss / total_samples
+                val_loss = self.eval()
+                row = {
+                    "epoch": ep + 1,
+                    "train_mse": train_loss,
+                    "val_mse": val_loss,
+                    "lr": learning_rate,
+                }
+                self.history.append(row)
+                writer.writerow(row)
+                stream.flush()
+                self.lr_scheduler.step()
+
+                if (ep + 1) % self.args.save_epoch == 0:
+                    torch.save(
+                        self.model.state_dict(),
+                        self.checkpoint_dir / f"epoch_{ep + 1}.pt",
+                    )
+                if ep == 0 or (ep + 1) % self.args.log_every == 0 or ep + 1 == self.args.epoch:
+                    print(
+                        f"Epoch {ep + 1}/{self.args.epoch} | "
+                        f"train MSE={train_loss:.8f} | val MSE={val_loss:.8f} | "
+                        f"lr={learning_rate:.6g}",
+                        flush=True,
+                    )
+
+        torch.save(self.model.state_dict(), self.checkpoint_path)
+        self._save_loss_curve()
+
+    def _evaluate(self, loader):
         self.model.eval()
-        eval_loss = 0.0
-        with torch.no_grad():
-            for batch in self.eval_loader:
-                x,y = batch
-                yx = self.model(x)
-                loss = self.loss_fun(y,yx)
-                batch_size = x.size(0)
-                eval_loss += loss.item() * batch_size
-                total_samples += batch_size
-                
-        eval_loss /= total_samples
-        
-    def test(self):
-        checkpoint = './checkpoints'+self.name  + '.pt'
-        with open(checkpoint,'rb') as stream:
-            self.model.load_state_dict(torch.load(stream))
-        self.model.eval()
-        test_loss = 0.0
+        total_loss = 0.0
         total_samples = 0
         with torch.no_grad():
-            for batch in self.test_loader:
-                x,y = batch
-                yx = self.model(x)
-                loss = self.loss_fun(y,yx)
+            for x, y in loader:
+                x = x.to(self.device)
+                y = y.to(self.device)
+                prediction = self.model(x)
+                loss = self.loss_fun(prediction, y)
                 batch_size = x.size(0)
-                test_loss += loss.item() * batch_size
+                total_loss += loss.item() * batch_size
                 total_samples += batch_size
-        test_loss /= total_samples+        return total_loss / total_samples
+
+    def eval(self):
+        return self._evaluate(self.eval_loader)
+
+    def test(self):
+        state_dict = torch.load(
+            self.checkpoint_path, map_location=self.device, weights_only=True
+        )
+        self.model.load_state_dict(state_dict)
+        test_mse = self._evaluate(self.test_loader)
+        print(f"Test MSE: {test_mse:.10f}", flush=True)
+        return test_mse
+
+    def _save_loss_curve(self):
+        epochs = [row["epoch"] for row in self.history]
+        fig, ax = plt.subplots(figsize=(8, 5))
+        ax.plot(epochs, [row["train_mse"] for row in self.history], label="Training MSE")
+        ax.plot(epochs, [row["val_mse"] for row in self.history], label="Validation MSE")
+        ax.set(xlabel="Epoch", ylabel="MSE", title="Feedforward network fitting sin(x)")
+        ax.set_yscale("log")
+        ax.legend()
+        ax.grid(True, alpha=0.3)
+        fig.tight_layout()
+        fig.savefig(self.output_dir / "loss_curve.png", dpi=160)
+        plt.close(fig)
--- before/src/dataset/gen_datasets.py
+++ after/src/dataset/gen_datasets.py
@@ -5,7 +5,7 @@
 import math
 
 INT_MAX = 10000000000
-PI  = 3.14
+PI = math.pi
 def gen_data(train_num:int,eval_num:int,test_num:int):
     # follow the function y = sin(x) x \in (0,2pi)
     total_num = train_num + eval_num + test_num
--- before/requirements.txt
+++ after/requirements.txt
@@ -1 +1,4 @@
-PyYAML>=6.0,<7.0
+torch==2.8.0
+numpy==2.5.4
+PyYAML==6.0.3
+matplotlib==3.11.2
--- before/.gitignore
+++ after/.gitignore
@@ -6,3 +6,6 @@
 .env
 .env.*
 !.env.example
+
+# Generated experiment results and model parameters
+outputs/
```
