#%% [markdown]
# # 世界人口回归

#%%
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.optim

#%% [markdown]
# 读取数据

#%%
url = r'http://en.wikipedia.org/wiki/World_population_estimates'
df = pd.read_html(url, header=0, attrs={"class" : "wikitable"})[2]
df


#%%
years = torch.tensor(df.iloc[:, 0], dtype=torch.float32)
populations = torch.tensor(df.iloc[:, 1], dtype=torch.float32)

#%% [markdown]
# 线性回归

#%%
x = torch.stack([years, torch.ones_like(years)], 1)
y = populations
wr, _ = torch.gels(y, x)
slope, intercept = wr[:2, 0]
result = 'population = {:.2e} * year + {:.2e}'.format(slope, intercept)
print('回归结果：' + result)


#%%
x = torch.stack([years, torch.ones_like(years)], 1)
y = populations
w = x.t().mm(x).inverse().mm(x.t()).mv(y)
slope, intercept = w
result = 'population = {:.2e} * year + {:.2e}'.format(slope, intercept)
print('回归结果：' + result)


#%%
plt.scatter(years, populations, s=0.1, label='actual', color='k')
plt.plot(years.tolist(), (slope * years + intercept).tolist(), label=result, color='k')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.show();


#%%
x = years.reshape(-1, 1)
y = populations

x_mean, x_std = torch.mean(x), torch.std(x)
x_norm = (x - x_mean) / x_std
y_mean, y_std = torch.mean(y), torch.std(y)
y_norm = (y - y_mean) / y_std

fc = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(fc.parameters())
weight_norm, bias_norm = fc.parameters()

for step in range(5001):
    if step:
        fc.zero_grad()
        loss_norm.backward()
        optimizer.step()
    output_norm = fc(x_norm)
    pred_norm = output_norm.squeeze()
    loss_norm = criterion(pred_norm, y_norm)
    weight = y_std / x_std * weight_norm
    bias = (weight_norm * (0 - x_mean) / x_std + bias_norm) * y_std + y_mean
    if step % 1000 == 0:
        print('第{}步：weight = {}, bias = {}'.format(step, weight.item(), bias.item()))

result = 'population = {:.2e} * year + {:.2e}'.format(weight.item(), bias.item())
print('回归结果：' + result)


