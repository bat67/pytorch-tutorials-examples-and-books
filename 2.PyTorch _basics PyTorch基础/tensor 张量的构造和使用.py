#%% [markdown]
# # 张量的构造和使用

#%%
import torch

#%% [markdown]
# 构造张量

#%%
t1 = torch.tensor([0., 1., 2.])
t2 = torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
t3 = torch.tensor([[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
        [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]]])


#%%
t1 = torch.empty(2) # 未初始化
t2 = torch.zeros(2, 2) # 各元素值.
t3 = torch.ones(2, 2, 2) # 各元素值为1.
t4 = torch.full((2, 2, 2, 2), 3.) # 各元素值为3.


#%%
t2 = torch.empty(2, 2)
t2[0, 0] = 0.
t2[0, 1] = 1.
t2[1, 0] = 2.
t2[1, 1] = 3.
print(t2)
print(t2.equal(torch.tensor([[0., 1.], [2., 3.]])))


#%%
torch.zeros(2, 3, 4)
torch.ones(2, 3, 4)


#%%
torch.ones_like(t2)


#%%
torch.linspace(0, 3, steps=4)

#%% [markdown]
# 张量的性质

#%%
print('data = {}'.format(t2))
print('size = {}'.format(t2.size()))
print('dim = {}'.format(t2.dim()))
print('numel = {}'.format(t2.numel()))


#%%
t2.dtype

#%% [markdown]
# 改变张量的大小

#%%
tc = torch.arange(12) # 张量大小 (12,)
print('tc = {}'.format(tc))
t322 = tc.reshape(3, 2, 2) # 张量大小 (3, 2, 2)
print('t322 = {}'.format(t322))
t43 = t322.reshape(4, 3) # 张量大小 (4, 3)
print('t43 = {}'.format(t43))


#%%
t12 = torch.tensor([[5., -9.],])
t21 = t12.transpose(0, 1)
print('t21 = {}'.format(t21))
t21 = t12.t()
print('t21 = {}'.format(t21))


#%%
t12 = torch.tensor([[5., -9.],])
print('t12 = {}'.format(t12))
t34 = t12.repeat(3, 2)
print('t34 = {}'.format(t34))


#%%
t44 = torch.arange(16).reshape(4, 4)
print('t44 = {}'.format(t44))
t23 = t44[1:-1, :3]
print('t23 = {}'.format(t23))

#%% [markdown]
# 张量的数学运算

#%%
tl = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
tr = torch.tensor([[7., 8., 9.], [10., 11., 12.]])
print(tl + tr) # 加法
print(tl - tr) # 减法
print(tl * tr) # 乘法
print(tl / tr) # 除法
print(tl ** tr) # 有理数次乘方
print(tl ** (1 / tr)) # 有理数次开方


#%%
print(torch.zeros(3, 4) + 5) # 得到各元素全为5的大小为(3,4)的张量
print(-6 * torch.ones(2)) # 得到各元素全为-6的大小为(2,)的张量
print(torch.ones(2, 3, 4) + torch.ones(4)) # 得到各元素全为2的大小为(2,3,4)的张量


#%%
t234 = torch.arange(24).reshape(2, 3, 4)
print('sqrt = {}'.format(t234.sqrt()))
print('sum = {}'.format(t234.sum()))
print('prod = {}'.format(t234.prod()))
print('norm(2) = {}'.format(t234.norm(2)))
print('cumsum = {}'.format(t234.cumsum(dim=0)))
print('cumprod = {}'.format(t234.cumprod(dim=1)))


#%%
tp = torch.pow(torch.arange(1, 4), torch.arange(3))
print('pow = {}'.format(tp))
te = torch.exp(torch.tensor([0.1, -0.01]))
print('exp = {}'.format(te))
ts = torch.sin(torch.tensor([[3.14 / 4,],]))
print('sin = {}'.format(ts))


#%%
t5 = torch.arange(5)
tf = torch.frac(t5 * 0.3)
print('frac = {}'.format(tf))
tc = torch.clamp(t5, 0.5, 3.5)
print('clamp = {}'.format(tc))

#%% [markdown]
# 张量的拼接

#%%
tp = torch.arange(12).reshape(3, 4)
tn = -tp
tc0 = torch.cat([tp, tn], 0)
print('tc0 = {}'.format(tc0))
tc1 = torch.cat([tp, tp, tn, tn], 1)
print('tc1 = {}'.format(tc1))


#%%
tp = torch.arange(12).reshape(3, 4)
tn = -tp
ts0 = torch.stack([tp, tn], 0)
print('ts0 = {}'.format(ts0))
ts1 = torch.stack([tp, tp, tn, tn], 1)
print('ts1 = {}'.format(ts1))


