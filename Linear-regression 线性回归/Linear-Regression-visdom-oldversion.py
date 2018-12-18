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


#%%
# visdom is a visualization tool from facebook

from visdom import Visdom
viz = Visdom()

#%% [markdown]
# ## 2. Generate Data

#%%
num_data = 1000 
num_epoch = 1000

noise = init.normal(torch.FloatTensor(num_data,1),std=0.2)
x = init.uniform(torch.Tensor(num_data,1),-10,10)
y = 2*x+3
y_noise = 2*(x+noise)+3


#%%
# visualize data with visdom

input_data = torch.cat([x,y_noise],1)

win=viz.scatter(
    X = input_data,
    opts=dict(
        xtickmin=-10,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-20,
        ytickmax=20,
        ytickstep=1,
        markersymbol='dot',
        markersize=5,
        markercolor=np.random.randint(0, 255, num_data),
    ),
)

viz.updateTrace(
    X = x,
    Y = y,
    win=win,
)

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

#%% [markdown]
# ## 6. Visualize output

#%%
win_2=viz.scatter(
    X = input_data,
    opts=dict(
        xtickmin=-10,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-20,
        ytickmax=20,
        ytickstep=1,
        markersymbol='dot',
        markercolor=np.random.randint(0, 255, num_data),
        markersize=5,
    ),
)

viz.updateTrace(
    X = x,
    Y = output.data,
    win = win_2,
    opts=dict(
        xtickmin=-15,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-300,
        ytickmax=200,
        ytickstep=1,
        markersymbol='dot',
    ),
)

#%% [markdown]
# ## 7. Visualize Loss Graph

#%%
x = np.reshape([i for i in range(num_epoch)],newshape=[num_epoch,1])
loss_data = np.reshape(loss_arr,newshape=[num_epoch,1])

win2=viz.line(
    X = x,
    Y = loss_data,
    opts=dict(
        xtickmin=0,
        xtickmax=num_epoch,
        xtickstep=1,
        ytickmin=0,
        ytickmax=20,
        ytickstep=1,
        markercolor=np.random.randint(0, 255, num_epoch),
    ),
)


