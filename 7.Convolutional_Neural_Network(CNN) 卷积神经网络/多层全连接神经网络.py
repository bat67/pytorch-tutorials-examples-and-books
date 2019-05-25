#多层全连接神经网络
import torch
from torch import nn,optim
from torch.utils.data import dataloader


#第一种 简单的全连接的神经网络
class simpleNet(nn.Module):

    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet,self).__init__()
        self.layer1=nn.Linear(in_dim,n_hidden_1)
        self.layer2=nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3=nn.Linear(n_hidden_2,out_dim)

    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

#第二种添加激活函数，增加网络的非线性
class Actication_Net(nn.Module):

    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Actication_Net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim,n_hidden_1),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1,n_hidden_2),
            nn.ReLU(True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2,out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#第三种 添加批处理标准化 ，加快收敛的速度
class Batch_Net(nn.Module):

    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):

        super(Batch_Net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim,n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1,n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2,out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


