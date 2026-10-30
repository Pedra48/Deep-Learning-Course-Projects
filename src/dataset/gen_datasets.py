import torch
from torch.utils.data import TensorDataset
import random
import numpy as np
import math

INT_MAX = 10000000000
PI = math.pi
def gen_data(train_num:int,eval_num:int,test_num:int):
    # follow the function y = sin(x) x \in (0,2pi)
    total_num = train_num + eval_num + test_num
    
    x = np.array(random.sample(range(1,INT_MAX),total_num))
    x = x * 2 * PI / INT_MAX 

    # train_dataset
    x_train = x[:train_num]
    y_train = np.sin(x_train) 
    x_train = torch.tensor(x_train,dtype=torch.float32).reshape(-1,1)
    y_train = torch.tensor(y_train,dtype=torch.float32).reshape(-1,1)
    train_dataset = TensorDataset(x_train, y_train)
    # eval_dataset
    x_eval = x[train_num:train_num+eval_num]
    y_eval = np.sin(x_eval)
    x_eval = torch.tensor(x_eval,dtype=torch.float32).reshape(-1,1)
    y_eval = torch.tensor(y_eval,dtype=torch.float32).reshape(-1,1)
    eval_dataset = TensorDataset(x_eval, y_eval)
    # test_dataset
    x_test = x[train_num+eval_num:]
    y_test = np.sin(x_test)
    x_test = torch.tensor(x_test,dtype=torch.float32).reshape(-1,1)
    y_test = torch.tensor(y_test,dtype=torch.float32).reshape(-1,1)
    test_dataset = TensorDataset(x_test, y_test)
    
    return train_dataset,eval_dataset,test_dataset
