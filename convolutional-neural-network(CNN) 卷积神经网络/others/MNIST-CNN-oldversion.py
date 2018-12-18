
#%%
# MNIST CNN classifier 
# Code by GunhoChoi

import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Set Hyperparameters

epoch = 100
batch_size =16
learning_rate = 0.001

# Download Data

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)


#%%
# Check the datasets downloaded

print(mnist_train.__len__())
print(mnist_test.__len__())
img1,label1 = mnist_train.__getitem__(0)
img2,label2 = mnist_test.__getitem__(0)

print(img1.size(), label1)
print(img2.size(), label2)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=batch_size,shuffle=True)


#%%
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
#                 padding=0, dilation=1, groups=1, bias=True)
# torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
#                    return_indices=False, ceil_mode=False)
# torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1,affine=True)
# torch.nn.ReLU()
# tensor.view(newshape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,16,5),   # batch x 16 x 24 x 24
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        nn.Conv2d(16,32,5),  # batch x 32 x 20 x 20
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.MaxPool2d(2,2)   # batch x 32 x 10 x 10
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32,64,5),  # batch x 64 x 6 x 6
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,128,5),  # batch x 128 x 2 x 2
                        nn.ReLU()
        )
        self.fc = nn.Linear(2*2*128,10)
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out
        
cnn = CNN()

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)


#%%
# Train Model with train data
# In order to use GPU you need to move all Variables and model by Module.cuda()

for i in range(epoch):
    for j,[image,label] in enumerate(train_loader):
        image = Variable(image)
        label = Variable(label)
        
        optimizer.zero_grad()
        result = cnn.forward(image)
        loss = loss_func(result,label)
        loss.backward()
        optimizer.step()
        
        if j % 100 == 0:
            print(loss)


#%%
# Test with test data
# In order test, we need to change model mode to .eval()
# and get the highest score label for accuracy

cnn.eval()
correct = 0
total = 0

for image,label in test_loader:
    image = Variable(image)
    result = cnn(image)
    
    _,pred = torch.max(result.data,1)
    
    total += label.size(0)
    correct += (pred == label).sum()
    
print("Accuracy of Test Data: {}".format(correct/total))


