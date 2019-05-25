#%% [markdown]
# # 学习率衰减
# 对于基于一阶梯度进行优化的方法而言，开始的时候更新的幅度是比较大的，也就是说开始的学习率可以设置大一点，但是当训练集的 loss 下降到一定程度之后，，使用这个太大的学习率就会导致 loss 一直来回震荡，比如
# 
# ![](https://ws4.sinaimg.cn/large/006tNc79ly1fmrvdlncomj30bf0aywet.jpg)
#%% [markdown]
# 这个时候就需要对学习率进行衰减已达到 loss 的充分下降，而是用学习率衰减的办法能够解决这个矛盾，学习率衰减就是随着训练的进行不断的减小学习率。
# 
# 在 pytorch 中学习率衰减非常方便，使用 `torch.optim.lr_scheduler`，更多的信息可以直接查看[文档](http://pytorch.org/docs/0.3.0/optim.html#how-to-adjust-learning-rate)
# 
# 但是我推荐大家使用下面这种方式来做学习率衰减，更加直观，下面我们直接举例子来说明

#%%
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from utils import resnet
from torchvision import transforms as tfs
from datetime import datetime


#%%
net = resnet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-4)

#%% [markdown]
# 这里我们定义好了模型和优化器，可以通过 `optimizer.param_groups` 来得到所有的参数组和其对应的属性，参数组是什么意思呢？就是我们可以将模型的参数分成几个组，每个组定义一个学习率，这里比较复杂，一般来讲如果不做特别修改，就只有一个参数组
# 
# 这个参数组是一个字典，里面有很多属性，比如学习率，权重衰减等等，我们可以访问以下

#%%
print('learning rate: {}'.format(optimizer.param_groups[0]['lr']))
print('weight decay: {}'.format(optimizer.param_groups[0]['weight_decay']))

#%% [markdown]
# 所以我们可以通过修改这个属性来改变我们训练过程中的学习率，非常简单

#%%
optimizer.param_groups[0]['lr'] = 1e-5

#%% [markdown]
# 为了防止有多个参数组，我们可以使用一个循环

#%%
for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-1

#%% [markdown]
# 方法就是这样，非常简单，我们可以在任意的位置改变我们的学习率
# 
# 下面我们具体来看看学习率衰减的好处

#%%
def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#%%
# 使用数据增强
def train_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize(120),
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(96),
        tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

def test_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize(96),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

train_set = CIFAR10('C:/DATASETS/cifar10/', train=True, transform=train_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
valid_set = CIFAR10('C:/DATASETS/cifar10/', train=False, transform=test_tf)
valid_data = torch.utils.data.DataLoader(valid_set, batch_size=256, shuffle=False)

net = resnet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


#%%
train_losses = []
valid_losses = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
prev_time = datetime.now()
for epoch in range(30):
    if epoch == 20:
        set_learning_rate(optimizer, 0.01) # 80 次修改学习率为 0.01
    train_loss = 0
    net = net.train()
    for im, label in train_data:
        im = im.to(device)
        label = label.to(device)

        # forward
        output = net(im)
        loss = criterion(output, label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    valid_loss = 0
    valid_acc = 0
    net = net.eval()
    with torch.no_grad():
        for im, label in valid_data:
            im = im.to(device)
            label = label.to(device)
            output = net(im)
            loss = criterion(output, label)
            valid_loss += loss.item()
    epoch_str = (
        "Epoch %d. Train Loss: %f, Valid Loss: %f, "
        % (epoch, train_loss / len(train_data), valid_loss / len(valid_data)))
    prev_time = cur_time
    
    train_losses.append(train_loss / len(train_data))
    valid_losses.append(valid_loss / len(valid_data))
    print(epoch_str + time_str)

#%% [markdown]
# 下面我们画出 loss 曲线

#%%
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='valid')
plt.xlabel('epoch')
plt.legend(loc='best')

#%% [markdown]
# 这里我们只训练了 30 次，在 20 次的时候进行了学习率衰减，可以看 loss 曲线在 20 次的时候不管是 train loss 还是 valid loss，都有了一个陡降。
# 
# 当然这里我们只是作为举例，在实际应用中，做学习率衰减之前应该经过充分的训练，比如训练 80 次或者 100 次，然后再做学习率衰减得到更好的结果，有的时候甚至需要做多次学习率衰减

