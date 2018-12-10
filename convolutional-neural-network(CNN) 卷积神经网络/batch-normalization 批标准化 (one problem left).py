#%% [markdown]
# # 批标准化
# 在我们正式进入模型的构建和训练之前，我们会先讲一讲数据预处理和批标准化，因为模型训练并不容易，特别是一些非常复杂的模型，并不能非常好的训练得到收敛的结果，所以对数据增加一些预处理，同时使用批标准化能够得到非常好的收敛结果，这也是卷积网络能够训练到非常深的层的一个重要原因。
#%% [markdown]
# ## 数据预处理
# 目前数据预处理最常见的方法就是中心化和标准化，中心化相当于修正数据的中心位置，实现方法非常简单，就是在每个特征维度上减去对应的均值，最后得到 0 均值的特征。标准化也非常简单，在数据变成 0 均值之后，为了使得不同的特征维度有着相同的规模，可以除以标准差近似为一个标准正态分布，也可以依据最大值和最小值将其转化为 -1 ~ 1 之间，下面是一个简单的图示
# 
# ![](https://ws1.sinaimg.cn/large/006tKfTcly1fmqouzer3xj30ij06n0t8.jpg)
# 
# 这两种方法非常的常见，如果你还记得，前面我们在神经网络的部分就已经使用了这个方法实现了数据标准化，至于另外一些方法，比如 PCA 或者 白噪声已经用得非常少了。
#%% [markdown]
# ## Batch Normalization
# 前面在数据预处理的时候，我们尽量输入特征不相关且满足一个标准的正态分布，这样模型的表现一般也较好。但是对于很深的网路结构，网路的非线性层会使得输出的结果变得相关，且不再满足一个标准的 N(0, 1) 的分布，甚至输出的中心已经发生了偏移，这对于模型的训练，特别是深层的模型训练非常的困难。
# 
# 所以在 2015 年一篇论文提出了这个方法，批标准化，简而言之，就是对于每一层网络的输出，对其做一个归一化，使其服从标准的正态分布，这样后一层网络的输入也是一个标准的正态分布，所以能够比较好的进行训练，加快收敛速度。
#%% [markdown]
# batch normalization 的实现非常简单，对于给定的一个 batch 的数据 $B = \{x_1, x_2, \cdots, x_m\}$算法的公式如下
# 
# $$
# \mu_B = \frac{1}{m} \sum_{i=1}^m x_i
# $$
# $$
# \sigma^2_B = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2
# $$
# $$
# \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}
# $$
# $$
# y_i = \gamma \hat{x}_i + \beta
# $$
#%% [markdown]
# 第一行和第二行是计算出一个 batch 中数据的均值和方差，接着使用第三个公式对 batch 中的每个数据点做标准化，$\epsilon$ 是为了计算稳定引入的一个小的常数，通常取 $10^{-5}$，最后利用权重修正得到最后的输出结果，非常的简单，下面我们可以实现一下简单的一维的情况，也就是神经网络中的情况

#%%
import torch


#%%
def simple_batch_norm_1d(x, gamma, beta):
    eps = 1e-5
    x_mean = torch.mean(x, dim=0, keepdim=True) # 保留维度进行 broadcast
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

#%% [markdown]
# 我们来验证一下是否对于任意的输入，输出会被标准化

#%%
x = torch.arange(15.0).view(5, 3)
gamma = torch.ones(x.shape[1])
beta = torch.zeros(x.shape[1])
print('before bn: ')
print(x)
y = simple_batch_norm_1d(x, gamma, beta)
print('after bn: ')
print(y)

#%% [markdown]
# 可以看到这里一共是 5 个数据点，三个特征，每一列表示一个特征的不同数据点，使用批标准化之后，每一列都变成了标准的正态分布
# 
# 这个时候会出现一个问题，就是测试的时候该使用批标准化吗？
# 
# 答案是肯定的，因为训练的时候使用了，而测试的时候不使用肯定会导致结果出现偏差，但是测试的时候如果只有一个数据集，那么均值不就是这个值，方差为 0 吗？这显然是随机的，所以测试的时候不能用测试的数据集去算均值和方差，而是用训练的时候算出的移动平均均值和方差去代替
# 
# 下面我们实现以下能够区分训练状态和测试状态的批标准化方法

#%%
def batch_norm_1d(x, gamma, beta, is_training, moving_mean, moving_var, moving_momentum=0.1):
    eps = 1e-5
    x_mean = torch.mean(x, dim=0, keepdim=True) # 保留维度进行 broadcast
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    if is_training:
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        moving_mean[:] = moving_momentum * moving_mean + (1. - moving_momentum) * x_mean
        moving_var[:] = moving_momentum * moving_var + (1. - moving_momentum) * x_var
    else:
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

#%% [markdown]
# 下面我们使用上一节课将的深度神经网络分类 mnist 数据集的例子来试验一下批标准化是否有用

#%%
import numpy as np
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据
from torch.utils.data import DataLoader
from torch import nn


#%%
# 使用内置函数下载 mnist 数据集
train_set = mnist.MNIST('C:/DATASETS/mnist/', train=True)
test_set = mnist.MNIST('C:/DATASETS/mnist/', train=False)

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 数据预处理，标准化
    x = x.reshape((-1,)) # 拉平
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('C:/DATASETS/mnist/', train=True, transform=data_tf, download=True) # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('C:/DATASETS/mnist/', train=False, transform=data_tf, download=True)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)


#%%
class multi_network(nn.Module):
    def __init__(self):
        super(multi_network, self).__init__()
        self.layer1 = nn.Linear(784, 100)
        self.relu = nn.ReLU(True)
        self.layer2 = nn.Linear(100, 10)
        
        self.gamma = nn.Parameter(torch.randn(100))
        self.beta = nn.Parameter(torch.randn(100))
        
        self.moving_mean = torch.zeros(100)
        self.moving_var = torch.zeros(100)
        
    def forward(self, x, is_train=True):
        x = self.layer1(x)
        x = batch_norm_1d(x, self.gamma, self.beta, is_train, self.moving_mean, self.moving_var)
        x = self.relu(x)
        x = self.layer2(x)
        return x


#%%
net = multi_network()


#%%
net.parameters


#%%
# 定义 loss 函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1) # 使用随机梯度下降，学习率 0.1

#%% [markdown]
# 为了方便，训练函数已经定义在外面的 utils.py 中，跟前面训练网络的操作是一样的，感兴趣的同学可以去看看

#%%
from utils import train
train(net, train_data, test_data, 10, optimizer, criterion)

#%% [markdown]
# 这里的 $\gamma$ 和 $\beta$ 都作为参数进行训练，初始化为随机的高斯分布，`moving_mean` 和 `moving_var` 都初始化为 0，并不是更新的参数，训练完 10 次之后，我们可以看看移动平均和移动方差被修改为了多少

#%%
# 打出 moving_mean 的前 10 项
print(net.moving_mean[:10])

#%% [markdown]
# 可以看到，这些值已经在训练的过程中进行了修改，在测试过程中，我们不需要再计算均值和方差，直接使用移动平均和移动方差即可
#%% [markdown]
# 作为对比，我们看看不使用批标准化的结果

#%%
no_bn_net = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(True),
    nn.Linear(100, 10)
)

optimizer = torch.optim.SGD(no_bn_net.parameters(), 1e-1) # 使用随机梯度下降，学习率 0.1
train(no_bn_net, train_data, test_data, 10, optimizer, criterion)

#%% [markdown]
# 可以看到虽然最后的结果两种情况一样，但是如果我们看前几次的情况，可以看到使用批标准化的情况能够更快的收敛，因为这只是一个小网络，所以用不用批标准化都能够收敛，但是对于更加深的网络，使用批标准化在训练的时候能够很快地收敛
#%% [markdown]
# 从上面可以看到，我们自己实现了 2 维情况的批标准化，对应于卷积的 4 维情况的标准化是类似的，只需要沿着通道的维度进行均值和方差的计算，但是我们自己实现批标准化是很累的，pytorch 当然也为我们内置了批标准化的函数，一维和二维分别是 `torch.nn.BatchNorm1d()` 和 `torch.nn.BatchNorm2d()`，不同于我们的实现，pytorch 不仅将 $\gamma$ 和 $\beta$ 作为训练的参数，也将 `moving_mean` 和 `moving_var` 也作为参数进行训练
#%% [markdown]
# 下面我们在卷积网络下试用一下批标准化看看效果

#%%
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 数据预处理，标准化
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True) # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)


#%%
# 使用批标准化
class conv_bn_net(nn.Module):
    def __init__(self):
        super(conv_bn_net, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classfy = nn.Linear(400, 10)
    def forward(self, x):
        x = self.stage1(x)
        x = x.view(x.shape[0], -1)
        x = self.classfy(x)
        return x

net = conv_bn_net()
optimizer = torch.optim.SGD(net.parameters(), 1e-1) # 使用随机梯度下降，学习率 0.1


#%%
train(net, train_data, test_data, 5, optimizer, criterion)


#%%
# 不使用批标准化
class conv_no_bn_net(nn.Module):
    def __init__(self):
        super(conv_no_bn_net, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classfy = nn.Linear(400, 10)
    def forward(self, x):
        x = self.stage1(x)
        x = x.view(x.shape[0], -1)
        x = self.classfy(x)
        return x

net = conv_no_bn_net()
optimizer = torch.optim.SGD(net.parameters(), 1e-1) # 使用随机梯度下降，学习率 0.1    


#%%
train(net, train_data, test_data, 5, optimizer, criterion)

#%% [markdown]
# 之后介绍一些著名的网络结构的时候，我们会慢慢认识到批标准化的重要性，使用 pytorch 能够非常方便地添加批标准化层

