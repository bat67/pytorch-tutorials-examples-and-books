
#%%
# How to use Pretrained models with PyTorch
# Simple Classifier using resnet50
# code by GunhoChoi

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

batch_size = 3
learning_rate =0.0002
epoch = 50

resnet = models.resnet50(pretrained=True)


#%%
# Input pipeline from a folder containing multiple folders of images
# we can check the classes, class_to_idx, and filename with idx

img_dir = "./images"
img_data = dset.ImageFolder(img_dir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]))

print(img_data.classes)
print(img_data.class_to_idx)
print(img_data.imgs)


#%%
# After we get the list of images, we can turn the list into batches of images
# with torch.utils.data.DataLoader()

img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)

for img,label in img_batch:
    print(img.size())
    print(label)


#%%
# test of the result coming from resnet model

img = Variable(img)
print(resnet(img))


#%%
# we have 2 categorical variables so 1000 -> 500 -> 2
# test the whole process

model = nn.Sequential(
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500,2),
            nn.ReLU()
            )

print(model(resnet(img)))


#%%
# define loss func & optimizer

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


#%%
# In order to train with GPU, we need to put the model and variables
# by doing .cuda()

resnet.cuda()
model.cuda()

for i in range(epoch):
    for img,label in img_batch:
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        optimizer.zero_grad()
        output = model(resnet(img))
        loss = loss_func(output,label)
        loss.backward()
        optimizer.step()

    if i % 10 ==0:
        print(loss)


#%%
# Check Accuracy of the trained model
# Need to get used to using .cuda() and .data 

model.eval()
correct = 0
total = 0

for img,label in img_batch:
    img = Variable(img).cuda()
    label = Variable(label).cuda()
    
    output = model(resnet(img))
    _, pred = torch.max(output.data,1)
    
    total += label.size(0)
    correct += (pred == label.data).sum()   

print("Accuracy: {}".format(correct/total))


