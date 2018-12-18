#%% [markdown]
# # Linear Regression
# 
# - Linear Data
# - Linear Model
#%% [markdown]
# ## 1. Import Required Libraries

#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

#%% [markdown]
# ## 2. Generate Data

#%%
num_data = 1000 
num_epoch = 1000

noise = init.normal(torch.FloatTensor(num_data,1),std=0.2)
x = init.uniform(torch.Tensor(num_data,1),-10,10)
y = 2*x+3
y_noise = 2*(x+noise)+3

#%% [markdown]
# ## 3. Model & Optimizer

#%%
model = nn.Linear(1,1)
output = model(Variable(x))

loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

#%% [markdown]
# ## 4. Train

#%%
# train
loss_arr =[]
label = Variable(y_noise)
for i in range(num_epoch):
    output = model(Variable(x))
    optimizer.zero_grad()

    loss = loss_func(output,label)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(loss)
    loss_arr.append(loss.data.numpy()[0])

#%% [markdown]
# ## 5. Check Trained Parameters

#%%
param_list = list(model.parameters())
print(param_list[0].data,param_list[1].data)


