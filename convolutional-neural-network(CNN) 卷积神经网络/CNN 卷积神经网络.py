#%% [markdown]
# # 卷积神经网络

#%%
import torch
import torch.nn as nn

#%% [markdown]
# 卷积层

#%%
conv = torch.nn.Conv2d(16, 33, kernel_size=(3, 5), 
        stride=(2, 1), padding=(4, 2), dilation=(3, 1))
inputs = torch.randn(20, 16, 50, 100)
outputs = conv(inputs)
outputs.size()

#%% [markdown]
# 池化层

#%%
pool = nn.MaxPool1d(kernel_size=2, stride=2)
inputs = torch.randn(20, 16, 100)
outputs = pool(inputs)
outputs.size()


#%%
inputs = torch.tensor([[[1, 2, 3, 4, 5]]], dtype=torch.float32)
pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
outputs, indices = pool(inputs)
unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
recovers = unpool(outputs, indices)
recovers


#%%
inputs = torch.tensor([[[1, 2, 3, 4, 5]]], dtype=torch.float32)
pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
outputs, indices = pool(inputs)
unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
recovers = unpool(outputs, indices, output_size=inputs.size())
recovers


#%%
pool = nn.MaxPool1d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool1d(2, stride=2)
inputs = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]], dtype=torch.float32)
outputs, indices = pool(inputs)
unpool(outputs, indices)

#%% [markdown]
# 上采样层

#%%
inputs = torch.tensor([[[1, 0, 2]]], dtype=torch.float32)
# m = nn.Upsample(scale_factor=3, mode='nearest')
m = nn.Upsample(scale_factor=3, mode='linear')
inputs, m(inputs)


#%%
inputs = torch.tensor([[[1, 0, 2]]], dtype=torch.float32)
# m = nn.Upsample(size=9, mode='nearest')
m = nn.Upsample(size=8, mode='linear')
inputs, m(inputs), m(inputs) * 7


#%%
inputs = torch.arange(0, 4).reshape(1, 1, 4)
inputs = torch.tensor([[[1, 0, 2]]], dtype=torch.float32)
# m = nn.Upsample(scale_factor=2, mode='nearest')
m = nn.Upsample(scale_factor=2, mode='linear')
inputs, m(inputs)


#%%
inputs = torch.arange(1, 5).view(1, 1, 2, 2)
upsample = nn.Upsample(scale_factor=2, mode='nearest')
upsample(inputs)

#%% [markdown]
# 补全层

#%%
inputs = torch.arange(25).view(1, 1, 5, 5)
pad = nn.ConstantPad2d([1, 1, 1, 1], value=-1)
outputs = pad(inputs)
inputs, outputs


#%%
inputs = torch.arange(25).view(1, 1, 5, 5)
pad = nn.ReplicationPad2d([1, 1, 1, 1])
outputs = pad(inputs)
inputs, outputs


#%%
inputs = torch.arange(25).view(1, 1, 5, 5)
pad = nn.ReflectionPad2d([1, 1, 1, 1])
outputs = pad(inputs)
inputs, outputs


#%%
inputs = torch.arange(12).view(1, 1, 3, 4)
pad = nn.ConstantPad2d(padding=[1, 1, 1, 1], value=-1)
print ('常数补全 = {}'.format(pad(inputs)))
pad = nn.ReplicationPad2d(padding=[1, 1, 1, 1])
print ('重复补全 = {}'.format(pad(inputs)))
pad = nn.ReflectionPad2d(padding=[1, 1, 1, 1])
print ('反射补全 = {}'.format(pad(inputs)))


