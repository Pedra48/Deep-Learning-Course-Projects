# train eval test
import torch
import torch.nn as nn


from .dataset.gen_datasets import gen_data
from .model import Model
from .utils.loss import Myloss
class Exp():
    def __init__(self,args):
        self.args = args
        # 提取 Data
        self.train_dataset,self.eval_dataset,self.test_dataset = gen_data(args.train_num,args.eval_num,args.test_num)
        
        # 初始化 Model
        model = Model(args)
        
    def train(self):
        # 1. 优化器设置
        self.optimizer = torch.optim.AdamW()
        # 2. 学习率调节器设置
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size= self.args.step_size,
            gamma= self.args.gamma
        )
        # 3. 损失函数设置
        self.loss_fun = Myloss()
        # 4. 开始训练
        for _ in self.args.epoch:
            
            # 4.1 提取data
            
            # 4.2 forward
            # 4.3 计算损失
            # 4.4 更新优化器并backward
            # 4.5 优化器和学习率调节器step in
    def eval():
        
    def test():