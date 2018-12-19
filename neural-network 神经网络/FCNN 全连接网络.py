#%% [markdown]
# # 全连接网络

#%%
import torch.nn

#%% [markdown]
# 前馈网络的搭建

#%%
net = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1),
        torch.nn.ReLU())
net

#%% [markdown]
# 非线性激活层用法

#%%
ac = torch.nn.Softmax(dim=1)
x = torch.tensor([[1., 2.],[3., 4.]], requires_grad=True)
ac(x)


#%%
ac = torch.nn.Softmax2d()
x = torch.arange(16, requires_grad=True).reshape(2, 2, 2, 2)
ac(x)


