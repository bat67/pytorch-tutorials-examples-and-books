#%% [markdown]
# # 正则化
# 前面我们讲了数据增强和 dropout，而在实际使用中，现在的网络往往不使用 dropout，而是用另外一个技术，叫正则化。
# 
# 正则化是机器学习中提出来的一种方法，有 L1 和 L2 正则化，目前使用较多的是 L2 正则化，引入正则化相当于在 loss 函数上面加上一项，比如
# 
# $$
# f = loss + \lambda \sum_{p \in params} ||p||_2^2
# $$
# 
# 就是在 loss 的基础上加上了参数的二范数作为一个正则化，我们在训练网络的时候，不仅要最小化 loss 函数，同时还要最小化参数的二范数，也就是说我们会对参数做一些限制，不让它变得太大。
#%% [markdown]
# 如果我们对新的损失函数 f 求导进行梯度下降，就有
# 
# $$
# \frac{\partial f}{\partial p_j} = \frac{\partial loss}{\partial p_j} + 2 \lambda p_j
# $$
# 
# 那么在更新参数的时候就有
# 
# $$
# p_j \rightarrow p_j - \eta (\frac{\partial loss}{\partial p_j} + 2 \lambda p_j) = p_j - \eta \frac{\partial loss}{\partial p_j} - 2 \eta \lambda p_j 
# $$
# 
#%% [markdown]
# 可以看到 $p_j - \eta \frac{\partial loss}{\partial p_j}$ 和没加正则项要更新的部分一样，而后面的 $2\eta \lambda p_j$ 就是正则项的影响，可以看到加完正则项之后会对参数做更大程度的更新，这也被称为权重衰减(weight decay)，在 pytorch 中正则项就是通过这种方式来加入的，比如想在随机梯度下降法中使用正则项，或者说权重衰减，`torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4)` 就可以了，这个 `weight_decay` 系数就是上面公式中的 $\lambda$，非常方便
# 
# 注意正则项的系数的大小非常重要，如果太大，会极大的抑制参数的更新，导致欠拟合，如果太小，那么正则项这个部分基本没有贡献，所以选择一个合适的权重衰减系数非常重要，这个需要根据具体的情况去尝试，初步尝试可以使用 `1e-4` 或者 `1e-3` 
# 
# 下面我们在训练 cifar 10 中添加正则项

#%%
import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from utils import train, resnet
from torchvision import transforms as tfs


#%%
def data_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize(96),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

train_set = CIFAR10('./data', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
test_set = CIFAR10('./data', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

net = resnet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-4) # 增加正则项
criterion = nn.CrossEntropyLoss()


#%%
from utils import train
train(net, train_data, test_data, 20, optimizer, criterion)


