
#%%
import torch

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
# t_u = torch.tensor([32.9000, 57.2000, 59.0000, 82.4000, 51.8000, 46.4000, 37.4000, 24.8000, 42.8000, 55.4000, 69.8000])
# t_u = torch.tensor([30.3057, 56.7230, 57.9168, 79.4823, 52.3995, 47.3290, 37.7892, 25.9138, 43.6609, 56.5834, 71.3901])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])


#%%
def model(t_u, w, b):
    return w * t_u + b


#%%
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


#%%
w = torch.ones(1)
b = torch.zeros(1)

t_p = model(t_u, w, b)
t_p


#%%
loss = loss_fn(t_p, t_c)
loss


#%%
delta = 0.1

loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c) - loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)


#%%
learning_rate = 1e-2

w = w - learning_rate * loss_rate_of_change_w


#%%
loss_rate_of_change_b = (loss_fn(model(t_u, w, b + delta), t_c) - loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)

b = b - learning_rate * loss_rate_of_change_b


#%%
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c)
    return dsq_diffs


#%%
def model(t_u, w, b):
    return w * t_u + b


#%%
def dmodel_dw(t_u, w, b):
    return t_u


#%%
def dmodel_db(t_u, w, b):
    return 1.0


#%%
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dw = dloss_fn(t_p, t_c) * dmodel_dw(t_u, w, b)
    dloss_db = dloss_fn(t_p, t_c) * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])


#%%
params = torch.tensor([1.0, 0.0])

nepochs = 100

learning_rate = 1e-2

for epoch in range(nepochs):
    # forward pass
    w, b = params
    t_p = model(t_u, w, b)

    loss = loss_fn(t_p, t_c)
    print('Epoch %d, Loss %f' % (epoch, float(loss)))
    
    # backward pass
    grad = grad_fn(t_u, t_c, t_p, w, b)

    print('Params:', params)
    print('Grad:', grad)
    
    params = params - learning_rate * grad
    
params


#%%
params = torch.tensor([1.0, 0.0])

nepochs = 100

learning_rate = 1e-4

for epoch in range(nepochs):
    # forward pass
    w, b = params
    t_p = model(t_u, w, b)

    loss = loss_fn(t_p, t_c)
    print('Epoch %d, Loss %f' % (epoch, float(loss)))
    
    # backward pass
    grad = grad_fn(t_u, t_c, t_p, w, b)

    print('Params:', params)
    print('Grad:', grad)
    
    params = params - learning_rate * grad
    
params


#%%
t_un = 0.1 * t_u


#%%
params = torch.tensor([1.0, 0.0])

nepochs = 100

learning_rate = 1e-2

for epoch in range(nepochs):
    # forward pass
    w, b = params
    t_p = model(t_un, w, b)

    loss = loss_fn(t_p, t_c)
    print('Epoch %d, Loss %f' % (epoch, float(loss)))
    
    # backward pass
    grad = grad_fn(t_un, t_c, t_p, w, b)

    print('Params:', params)
    print('Grad:', grad)
    
    params = params - learning_rate * grad
    
params


#%%
params = torch.tensor([1.0, 0.0])

nepochs = 5000

learning_rate = 1e-2

for epoch in range(nepochs):
    # forward pass
    w, b = params
    t_p = model(t_un, w, b)

    loss = loss_fn(t_p, t_c)
    print('Epoch %d, Loss %f' % (epoch, float(loss)))
    
    # backward pass
    grad = grad_fn(t_un, t_c, t_p, w, b)

    params = params - learning_rate * grad
    
params


#%%
def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    sq_diffs = (t_p - t_c)**2
    return sq_diffs.mean()


#%%
params = torch.tensor([1.0, 0.0], requires_grad=True)

loss = loss_fn(model(t_u, *params), t_c)


#%%
params.grad is None


#%%
loss.backward()


#%%
params.grad


#%%
if params.grad is not None:
    params.grad.zero_()


#%%
def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    sq_diffs = (t_p - t_c)**2
    return sq_diffs.mean()

params = torch.tensor([1.0, 0.0], requires_grad=True)

nepochs = 5000

learning_rate = 1e-2

for epoch in range(nepochs):
    # forward pass
    t_p = model(t_un, *params)
    loss = loss_fn(t_p, t_c)

    print('Epoch %d, Loss %f' % (epoch, float(loss)))
    
    # backward pass
    if params.grad is not None:
        params.grad.zero_()

    loss.backward()

    #params.grad.clamp_(-1.0, 1.0)
    #print(params, params.grad)

    params = (params - learning_rate * params.grad).detach().requires_grad_()

params
#t_p = model(t_un, *params)
#t_p


#%%
import torch.optim as optim

dir(optim)


#%%
params = torch.tensor([1.0, 0.0], requires_grad=True)

learning_rate = 1e-5

optimizer = optim.SGD([params], lr=learning_rate)


#%%
t_p = model(t_u, *params)

loss = loss_fn(t_p, t_c)

loss.backward()

optimizer.step()

params


#%%
params = torch.tensor([1.0, 0.0], requires_grad=True)

learning_rate = 1e-2

optimizer = optim.SGD([params], lr=learning_rate)

t_p = model(t_un, *params)

loss = loss_fn(t_p, t_c)

optimizer.zero_grad()

loss.backward()

optimizer.step()

params


#%%
def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    sq_diffs = (t_p - t_c)**2
    return sq_diffs.mean()

params = torch.tensor([1.0, 0.0], requires_grad=True)

nepochs = 5000
learning_rate = 1e-2

optimizer = optim.SGD([params], lr=learning_rate)

for epoch in range(nepochs):
    
    # forward pass
    t_p = model(t_un, *params)
    loss = loss_fn(t_p, t_c)

    print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
    # backward pass
    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()

t_p = model(t_un, *params)

params


#%%
def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    sq_diffs = (t_p - t_c)**2
    return sq_diffs.mean()

params = torch.tensor([1.0, 0.0], requires_grad=True)

nepochs = 5000
learning_rate = 1e-1

optimizer = optim.Adam([params], lr=learning_rate)

for epoch in range(nepochs):
    # forward pass
    t_p = model(t_u, *params)
    loss = loss_fn(t_p, t_c)

    print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

t_p = model(t_u, *params)

params


#%%
from matplotlib import pyplot as plt

plt.plot(0.1 * t_u.numpy(), t_p.detach().numpy())
plt.plot(0.1 * t_u.numpy(), t_c.numpy(), 'o')


#%%
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_indices, val_indices


#%%
t_u_train = t_u[train_indices]
t_c_train = t_c[train_indices]

t_u_val = t_u[val_indices]
t_c_val = t_c[val_indices]


#%%
def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    sq_diffs = (t_p - t_c)**2
    return sq_diffs.mean()

params = torch.tensor([1.0, 0.0], requires_grad=True)

nepochs = 5000
learning_rate = 1e-2

optimizer = optim.SGD([params], lr=learning_rate)

t_un_train = 0.1 * t_u_train
t_un_val = 0.1 * t_u_val

for epoch in range(nepochs):
    
    # forward pass
    t_p_train = model(t_un_train, *params)
    loss_train = loss_fn(t_p_train, t_c_train)

    t_p_val = model(t_un_val, *params)
    loss_val = loss_fn(t_p_val, t_c_val)

    print('Epoch %d, Training loss %f, Validation loss %f' % (epoch, float(loss_train), float(loss_val)))
        
    # backward pass
    optimizer.zero_grad()
    loss_train.backward()    
    optimizer.step()

t_p = model(t_un, *params)

params


#%%
for epoch in range(nepochs):
    
    # forward pass
    t_p_train = model(t_un_train, *params)
    loss_train = loss_fn(t_p_train, t_c_train)

    with torch.no_grad():
        t_p_val = model(t_un_val, *params)
        loss_val = loss_fn(t_p_val, t_c_val)

    print('Epoch %d, Training loss %f, Validation loss %f' % (epoch, float(loss_train), float(loss_val)))
        
    # backward pass
    optimizer.zero_grad()
    loss_train.backward()    
    optimizer.step()


#%%
for epoch in range(nepochs):
    # ...
    print(loss_val.requires_grad)  # prints False
    # ...


#%%
def forward(t_u, t_c, is_train):
    with torch.set_grad_enabled(is_train):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
    return loss


#%%
import torch
import torch.nn as nn

model = nn.Linear(1, 1) # We'll look into the arguments in a minute


#%%
y = model.forward(x)  # DON'T DO THIS
y = model(x)          # DO THIS


#%%
model.weight


#%%
x = torch.ones(1)

model(x)


#%%
x = torch.ones(10, 1)

model(x)


#%%
t_u = torch.unsqueeze(t_u, 1)
t_c = torch.unsqueeze(t_c, 1)


#%%
model = nn.Linear(1, 1)


#%%
model.parameters()


#%%
list(model.parameters())


#%%
model = nn.Linear(1, 1)

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)


#%%
t_u_train = t_u[train_indices]
t_c_train = t_c[train_indices]

t_u_val = t_u[val_indices]
t_c_val = t_c[val_indices]

t_un_train = 0.1 * t_u_train
t_un_val = 0.1 * t_u_val


#%%
for epoch in range(nepochs):
    
    # forward pass
    t_p_train = model(t_un_train)
    loss_train = loss_fn(t_p_train, t_c_train)

    with torch.no_grad():
        t_p_val = model(t_un_val)
        loss_val = loss_fn(t_p_val, t_c_val)

    print('Epoch %d, Training loss %f, Validation loss %f' % (epoch, float(loss_train), float(loss_val)))
        
    # backward pass
    optimizer.zero_grad()
    loss_train.backward()    
    optimizer.step()


#%%
model.weight, model.bias


#%%
loss_fn = nn.MSELoss()


#%%
import torch
import torch.nn as nn

model = nn.Linear(1, 1)

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.MSELoss()

nepochs = 5000

for epoch in range(nepochs):
    # forward pass
    t_p_train = model(t_un_train)
    loss_train = loss_fn(t_p_train, t_c_train)

    with torch.no_grad():
        t_p_val = model(t_un_val)
        loss_val = loss_fn(t_p_val, t_c_val)

    print('Epoch %d, Training loss %f, Validation loss %f' % (epoch, float(loss_train), float(loss_val)))
        
    # backward pass
    optimizer.zero_grad()
    loss_train.backward()    
    optimizer.step()
    
model.weight, model.bias


#%%
model = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1))
model


#%%
[param.shape for param in model.parameters()]


#%%
for name, param in model.named_parameters():
    print(name, param.shape)


#%%
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 10)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(10, 1))
]))

model


#%%
for name, param in model.named_parameters():
    print(name, param.shape)


#%%
model.hidden_linear.weight


#%%
import torch
import torch.nn as nn

model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 10)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(10, 1))
]))

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.MSELoss()

nepochs = 5000

for epoch in range(nepochs):
    # forward pass
    t_p_train = model(t_un_train)
    loss_train = loss_fn(t_p_train, t_c_train)

    with torch.no_grad():
        t_p_val = model(t_un_val)
        loss_val = loss_fn(t_p_val, t_c_val)

    print('Epoch %d, Training loss %f, Validation loss %f' % (epoch, float(loss_train), float(loss_val)))
        
    # backward pass
    optimizer.zero_grad()
    loss_train.backward()    
    optimizer.step()
    
model(t_un_val), t_c_val


#%%
from matplotlib import pyplot as plt

plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_u.numpy(), model(0.1 * t_u).detach().numpy(), 'x')


#%%
from matplotlib import pyplot as plt

plt.plot(t_un.numpy(), t_c.numpy(), 'o')
plt.plot(t_un.numpy(), model(t_un, w, b).numpy(), 'r-')

#plt.plot(t_u.numpy(), model(0.1 * t_u).detach().numpy(), 'x')


