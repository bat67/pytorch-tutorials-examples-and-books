
#%%
from torchvision import models


#%%
dir(models)


#%%
alexnet = models.AlexNet()


#%%
resnet = models.resnet101(pretrained=True)


#%%
resnet


#%%
from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])


#%%
from PIL import Image
img = Image.open("bobby.jpg")


#%%
img


#%%
img_t = preprocess(img)


#%%
import torch
batch_t = torch.unsqueeze(img_t, 0)


#%%
resnet.eval()


#%%
out = resnet(batch_t)
out


#%%
with open('imagenet_classes.txt') as f:
    labels = f.readlines()


#%%
_, index = torch.max(out, 1)


#%%
labels[index[0]]


#%%
_, indices = torch.sort(out, descending=True)
[labels[idx] for idx in indices[0]][:5]


