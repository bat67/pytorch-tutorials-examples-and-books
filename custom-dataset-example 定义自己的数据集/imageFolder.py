from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

#加上transforms
normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
transform=transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])

dataset=ImageFolder('data/dogcat_2/',transform=transform)

#dataloader是一个可迭代的对象，意味着我们可以像使用迭代器一样使用它 或者 or batch_datas, batch_labels in dataloader:
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)

dataiter = iter(dataloader)
imgs, labels = next(dataiter)
print(imgs.size()) # batch_size, channel, height, weight
#输出 torch.Size([3, 3, 224, 224])
