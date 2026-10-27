import torch
import random
import numpy as np
import math

INT_MAX = 10000000000
PI  = 3.14
def gen_data(train_num:int,eval_num:int,test_num:int):
    # follow the function y = sin(x) x \in (0,2pi)
    total_num = train_num + eval_num + test_num
    
    x = np.array(random.sample(range(1,INT_MAX),total_num))
    x = x * 2 * PI / INT_MAX 

    # train_dataset
    x_train = x[:train_num]
    y_train = np.sin(x_train) 
    train_dataset = [x_train,y_train]
    # eval_dataset
    x_eval = x[train_num:train_num+eval_num]
    y_eval = np.sin(x_eval)
    eval_dataset = [x_eval,y_eval]
    # test_dataset
    x_test = x[train_num+eval_num:]
    y_test = np.sin(x_test)
    test_dataset = [x_test,y_test]
    
    return train_dataset,eval_dataset,test_dataset