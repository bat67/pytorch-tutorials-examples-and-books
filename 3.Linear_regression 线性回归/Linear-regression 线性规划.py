#%% [markdown]
# # 线性规划

#%%
import torch
import torch.nn
import torch.optim

#%% [markdown]
# 用 gels() 最小二乘

#%%
x = torch.tensor([[1., 1., 1.], [2., 3., 1.],
        [3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
y = torch.tensor([-10., 12., 14., 16., 18.])
wr, _ = torch.gels(y, x)
w = wr[:3]
w


#%%
x = torch.tensor([[1., 1., 1.], [2., 3., 1.], 
        [3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
y = torch.tensor([[-10., -3.], [12., 14.], [14., 12.], [16., 16.], [18., 16.]])
wr, _ = torch.gels(y, x)
w = wr[:3, :]
w

#%% [markdown]
# MSE 损失

#%%
criterion = torch.nn.MSELoss()
pred = torch.arange(5, requires_grad=True)
y = torch.ones(5)
loss = criterion(pred, y)
loss

#%% [markdown]
# 用优化引擎求解

#%%
x = torch.tensor([[1., 1., 1.], [2., 3., 1.], 
        [3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
y = torch.tensor([-10., 12., 14., 16., 18.])
w = torch.zeros(3, requires_grad=True)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam([w,],)

for step in range(30001):
    if step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    pred = torch.mv(x, w)
    loss = criterion(pred, y)
    if step % 1000 == 0:
        print('step = {}, loss = {:g}, W = {}'.format(
                step, loss, w.tolist()))


#%%
x = torch.tensor([[1., 1.], [2., 3.], [3., 5.], [4., 2.], [5., 4.]])
y = torch.tensor([-10., 12., 14., 16., 18.]).reshape(-1, 1)

fc = torch.nn.Linear(2, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(fc.parameters())
weights, bias = fc.parameters()

for step in range(30001):
    if step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    pred = fc(x)
    loss = criterion(pred, y)
    
    if step % 1000 == 0:
        print('step = {}, loss = {:g}, weights = {}, bias={}'.format(
                step, loss, weights[0, :].tolist(), bias.item()))

#%% [markdown]
# 归一化

#%%
x = torch.tensor([[1, 1, 1], [2, 3, 1], [3, 5, 1],
        [4, 2, 1], [5, 4, 1]], dtype=torch.float32)
y = torch.tensor([-10, 12, 14, 16, 18],
        dtype=torch.float32)

x_mean, x_std = torch.mean(x, dim=0), torch.std(x, dim=0)
x_mean[-1], x_std[-1] = 0, 1
x_norm = (x - x_mean) / x_std

#%% [markdown]
# 无归一化和有归一化的比较

#%%
x = torch.tensor([[1000000, 0.0001], [2000000, 0.0003],
        [3000000, 0.0005], [4000000, 0.0002], [5000000, 0.0004]])
y = torch.tensor([-1000., 1200., 1400., 1600., 1800.]).reshape(-1, 1)

fc = torch.nn.Linear(2, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(fc.parameters())

for step in range(10001):
    if step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pred = fc(x)
    loss = criterion(pred, y)
    if step % 1000 == 0:
        print('step = {}, loss = {:g}'.format(step, loss))


#%%
x = torch.tensor([[1000000, 0.0001], [2000000, 0.0003],
        [3000000, 0.0005], [4000000, 0.0002], [5000000, 0.0004]])
y = torch.tensor([-1000., 1200., 1400., 1600., 1800.]).reshape(-1, 1)

x_mean, x_std = torch.mean(x, dim=0), torch.std(x, dim=0)
x_norm = (x - x_mean) / x_std
y_mean, y_std = torch.mean(y, dim=0), torch.std(y, dim=0)
y_norm = (y - y_mean) / y_std

fc = torch.nn.Linear(2, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(fc.parameters())

for step in range(10001):
    if step:
        optimizer.zero_grad()
        loss_norm.backward()
        optimizer.step()
    pred_norm = fc(x_norm)
    loss_norm = criterion(pred_norm, y_norm)
    pred = pred_norm * y_std + y_mean
    loss = criterion(pred, y)
    if step % 1000 == 0:
        print('step = {}, loss = {:g}'.format(step, loss))


