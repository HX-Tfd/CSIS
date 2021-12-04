import torch.nn as nn
import torch
import torch.nn.functional as F

mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

'''
    computes the absolute sum of differences applied by {f} 
'''
class SumOfDifferences(nn.Module):
    def __init__(self, f):
        super().__init__()
        if f == 'l1':
            self.diff = nn.L1Loss()
        if f == 'l2':
            self.diff = nn.MSELoss()

    def forward(self, x, y):
        return self.diff(x, y) * len(x)