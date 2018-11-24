#%% [markdown]
# # 参数初始化
# 参数初始化对模型具有较大的影响，不同的初始化方式可能会导致截然不同的结果，所幸的是很多深度学习的先驱们已经帮我们探索了各种各样的初始化方式，所以我们只需要学会如何对模型的参数进行初始化的赋值即可。
#%% [markdown]
# PyTorch 的初始化方式并没有那么显然，如果你使用最原始的方式创建模型，那么你需要定义模型中的所有参数，当然这样你可以非常方便地定义每个变量的初始化方式，但是对于复杂的模型，这并不容易，而且我们推崇使用 Sequential 和 Module 来定义模型，所以这个时候我们就需要知道如何来自定义初始化方式
#%% [markdown]
# ## 使用 NumPy 来初始化
# 因为 PyTorch 是一个非常灵活的框架，理论上能够对所有的 Tensor 进行操作，所以我们能够通过定义新的 Tensor 来初始化，直接看下面的例子

#%%
import numpy as np
import torch
from torch import nn


#%%
# 定义一个 Sequential 模型
net1 = nn.Sequential(
    nn.Linear(30, 40),
    nn.ReLU(),
    nn.Linear(40, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)


#%%
# 访问第一层的参数
w1 = net1[0].weight
b1 = net1[0].bias


#%%
print(w1)

#%% [markdown]
# 注意，这是一个 Parameter，也就是一个特殊的 Tensor，我们可以访问其 `.data`属性得到其中的数据，然后直接定义一个新的 Tensor 对其进行替换，我们可以使用 PyTorch 中的一些随机数据生成的方式，比如 `torch.randn`，如果要使用更多 PyTorch 中没有的随机化方式，可以使用 numpy

#%%
# 定义一个 Tensor 直接对其进行替换
net1[0].weight.data = torch.from_numpy(np.random.uniform(3, 5, size=(40, 30)))


#%%
print(net1[0].weight)

#%% [markdown]
# 可以看到这个参数的值已经被改变了，也就是说已经被定义成了我们需要的初始化方式，如果模型中某一层需要我们手动去修改，那么我们可以直接用这种方式去访问，但是更多的时候是模型中相同类型的层都需要初始化成相同的方式，这个时候一种更高效的方式是使用循环去访问，比如

#%%
for layer in net1:
    if isinstance(layer, nn.Linear): # 判断是否是线性层
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=param_shape)) 
        # 定义为均值为 0，方差为 0.5 的正态分布

#%% [markdown]
# **小练习：一种非常流行的初始化方式叫 Xavier，方法来源于 2010 年的一篇论文 [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)，其通过数学的推到，证明了这种初始化方式可以使得每一层的输出方差是尽可能相等的，有兴趣的同学可以去看看论文**
# 
# 我们给出这种初始化的公式
# 
# $$
# w\ \sim \ Uniform[- \frac{\sqrt{6}}{\sqrt{n_j + n_{j+1}}}, \frac{\sqrt{6}}{\sqrt{n_j + n_{j+1}}}]
# $$
# 
# 其中 $n_j$ 和 $n_{j+1}$ 表示该层的输入和输出数目，所以请尝试实现以下这种初始化方式
#%% [markdown]
# 对于 Module 的参数初始化，其实也非常简单，如果想对其中的某层进行初始化，可以直接像 Sequential 一样对其 Tensor 进行重新定义，其唯一不同的地方在于，如果要用循环的方式访问，需要介绍两个属性，children 和 modules，下面我们举例来说明

#%%
class sim_net(nn.Module):
    def __init__(self):
        super(sim_net, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU()
        )
        
        self.l1[0].weight.data = torch.randn(40, 30) # 直接对某一层初始化
        
        self.l2 = nn.Sequential(
            nn.Linear(40, 50),
            nn.ReLU()
        )
        
        self.l3 = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


#%%
net2 = sim_net()


#%%
# 访问 children
for i in net2.children():
    print(i)


#%%
# 访问 modules
for i in net2.modules():
    print(i)

#%% [markdown]
# 通过上面的例子，看到区别了吗?
# 
# children 只会访问到模型定义中的第一层，因为上面的模型中定义了三个 Sequential，所以只会访问到三个 Sequential，而 modules 会访问到最后的结构，比如上面的例子，modules 不仅访问到了 Sequential，也访问到了 Sequential 里面，这就对我们做初始化非常方便，比如

#%%
for layer in net2.modules():
    if isinstance(layer, nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=param_shape)) 

#%% [markdown]
# 这上面实现了和 Sequential 相同的初始化，同样非常简便
#%% [markdown]
# ## torch.nn.init
# 因为 PyTorch 灵活的特性，我们可以直接对 Tensor 进行操作从而初始化，PyTorch 也提供了初始化的函数帮助我们快速初始化，就是 `torch.nn.init`，其操作层面仍然在 Tensor 上，下面我们举例说明

#%%
from torch.nn import init


#%%
print(net1[0].weight)


#%%
init.xavier_uniform_(net1[0].weight) # 这就是上面我们讲过的 Xavier 初始化方法，PyTorch 直接内置了其实现


#%%
print(net1[0].weight)

#%% [markdown]
# 可以看到参数已经被修改了
# 
# `torch.nn.init` 为我们提供了更多的内置初始化方式，避免了我们重复去实现一些相同的操作
#%% [markdown]
# 上面讲了两种初始化方式，其实它们的本质都是一样的，就是去修改某一层参数的实际值，而 `torch.nn.init` 提供了更多成熟的深度学习相关的初始化方式，非常方便

