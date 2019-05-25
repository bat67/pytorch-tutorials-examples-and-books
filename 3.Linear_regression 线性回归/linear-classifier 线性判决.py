#%% [markdown]
# # 线性判决

#%%
import torch
import torch.nn
import torch.optim

#%% [markdown]
# 两分类判决

#%%
x = torch.tensor([[1., 1., 1.], [2., 3., 1.],
        [3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
y = torch.tensor([0., 1., 1., 0., 1.])
w = torch.zeros(3, requires_grad=True)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam([w,],)

for step in range(100001):
    if step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pred = torch.mv(x, w)
    loss = criterion(pred, y)
    if step % 10000 == 0:
        print('第{}步：loss = {:g}, W = {}'.format(step, loss, w.tolist()))

#%% [markdown]
# 多分类判决

#%%
x = torch.tensor([[1., 1., 1.], [2., 3., 1.],
        [3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
y = torch.tensor([0, 2, 1, 0, 2])
w = torch.zeros(3, 3, requires_grad=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([w,],)

for step in range(100001):
    if step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pred = torch.mm(x, w)
    loss = criterion(pred, y)
    if step % 10000 == 0:
        print('第{}步：loss = {:g}, W = {}'.format(step, loss, w))


