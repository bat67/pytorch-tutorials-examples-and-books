#%% [markdown]
# # Himmelblau 函数的优化

#%%
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

import torch

#%% [markdown]
# 定义 Himmelblau 函数

#%%
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

#%% [markdown]
# 绘制 Himmelblau 函数

#%%
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
fig.show();

#%% [markdown]
# 求最小值

#%%

x = torch.tensor([0., 0.], requires_grad=True) # 收敛到 (3, 2)
# x = torch.tensor([-1., 0.], requires_grad=True) # 收敛到 (-2.81, 3.13)
# x = torch.tensor([-4., 0..], requires_grad=True) # 收敛到 (-3.78, -3.28)
# x = torch.tensor([4., 0.], requires_grad=True) # 收敛到 (3.58, -1.85)
optimizer = torch.optim.Adam([x,])
for step in range(20001):
    if step:
        optimizer.zero_grad()
        f.backward()
        optimizer.step()
    f = himmelblau(x)
    if step % 1000 == 0:
        print ('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), f))

#%% [markdown]
# 求极大值

#%%
x = torch.tensor([0., 0.], requires_grad=True)
optimizer = torch.optim.Adam([x,])
for step in range(20001):
    if step:
        optimizer.zero_grad()
        (-f).backward()
        optimizer.step()
    f = himmelblau(x)
    if step % 1000 == 0:
        print ('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), f))


