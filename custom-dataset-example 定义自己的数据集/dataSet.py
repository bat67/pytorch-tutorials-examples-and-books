#数据处理
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

transform=transforms.Compose([
    transforms.Resize(224), #缩放图片，保持长宽比不变，最短边的长为224像素,
    transforms.CenterCrop(224), #从中间切出 224*224的图片
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1]
])

#定义自己的数据集合
class DogCat(data.Dataset):

    def __init__(self,root,transform):
        #所有图片的绝对路径
        imgs=os.listdir(root)

        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=transform

    def __getitem__(self, index):
        img_path=self.imgs[index]
        #dog-> 1 cat ->0
        label=1 if 'dog' in img_path.split('/')[-1] else 0
        pil_img=Image.open(img_path)
        if self.transforms:
            data=self.transforms(pil_img)
        else:
            pil_img=np.asarray(pil_img)
            data=torch.from_numpy(pil_img)
        return data,label

    def __len__(self):
        return len(self.imgs)

dataSet=DogCat('./data/dogcat',transform=transform)

print(dataSet[0])
