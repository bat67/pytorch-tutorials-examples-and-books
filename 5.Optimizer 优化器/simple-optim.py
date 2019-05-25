#%% [markdown]
# # 梯度和优化

#%%
from math import pi
import torch
import torch.optim

#%% [markdown]
# 求梯度

#%%
x = torch.tensor([pi / 3,  pi / 6], requires_grad=True)
f = - ((x.cos() ** 2).sum()) ** 2
print('函数值 = {}'.format(f))
f.backward()
print('梯度值 = {}'.format(x.grad))
ref = 2 * (torch.cos(x) ** 2).sum() * torch.sin(2 * x)
print('梯度值(参考) = {}'.format(ref))

#%% [markdown]
# 优化问题求解

#%%
x = torch.tensor([pi / 3,  pi / 6], requires_grad=True)
optimizer = torch.optim.SGD([x,], lr=0.1, momentum=0)
for step in range(11):
    if step:
        optimizer.zero_grad()
        f.backward()
        optimizer.step()
    f = - ((x.cos() ** 2).sum()) ** 2
    print ('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), f))


