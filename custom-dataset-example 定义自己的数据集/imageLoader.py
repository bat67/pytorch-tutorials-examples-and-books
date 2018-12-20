'''
在数据处理中，有时会出现某个样本无法读取等问题，比如某张图片损坏。这时在__getitem__函数中将出现异常，此时最好的解决方案即是将出错的样本剔除。如果实在是遇到这种情况无法处理，则可以返回None对象，然后在Dataloader中实现自定义的collate_fn，将空对象过滤掉。但要注意，在这种情况下dataloader返回的batch数目会少于batch_size。
'''
from dataSet import *
import random
class NewDogCat(DogCat): # 继承前面实现的DogCat数据集
    def __getitem__(self, index):
        try:
            # 调用父类的获取函数，即 DogCat.__getitem__(self, index)
            return super(NewDogCat,self).__getitem__(index)
        except:
            #对于诸如样本损坏或数据集加载异常等情况，还可以通过其它方式解决。例如但凡遇到异常情况，就随机取一张图片代替：
            new_index = random.randint(0, len(self) - 1)
            return self[new_index]

from torch.utils.data.dataloader import default_collate # 导入默认的拼接方式
from torch.utils.data import DataLoader
def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x:x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据


transform=transforms.Compose([
    transforms.Resize(224), #缩放图片，保持长宽比不变，最短边的长为224像素,
    transforms.CenterCrop(224), #从中间切出 224*224的图片
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1]
])


dataset = NewDogCat(root='data/dogcat_wrong/', transform=transform)

#print(dataSet[11])
dataloader = DataLoader(dataset, 2, collate_fn=my_collate_fn, num_workers=1,shuffle=True)
for batch_datas, batch_labels in dataloader:
    print(batch_datas.size(),batch_labels.size())