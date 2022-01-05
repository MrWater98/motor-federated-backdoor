import numpy as np
import torch
from proprecess import prepro
from torch.utils.data import TensorDataset, DataLoader
def get_dataset(dir,conf):
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = prepro(d_path=dir,
                                                                conf=conf,
                                                                length=1024,
                                                                number=500,
                                                                normal=True,
                                                                rate=[0.7, 0.1, 0.2],
                                                                enc=False,
                                                                enc_step=28)
    
    train_X=train_X.reshape(-1,1,32,32)
    valid_X=valid_X.reshape(-1,1,32,32)
    test_X=test_X.reshape(-1,1,32,32)
    train_X=np.tile(train_X,[1,3,1,1])
    valid_X=np.tile(valid_X,[1,3,1,1])
    test_X=np.tile(test_X,[1,3,1,1])
    
    x =torch.tensor(train_X).type(torch.FloatTensor)
    y =torch.tensor(train_Y).type(torch.LongTensor)
    train_dataset = TensorDataset(x, y)
    
    x =torch.tensor(valid_X).type(torch.FloatTensor)
    y =torch.tensor(valid_Y).type(torch.LongTensor)
    eval_dataset = TensorDataset(x, y)
    
    return train_dataset, eval_dataset