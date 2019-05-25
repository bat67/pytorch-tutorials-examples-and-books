#%% [markdown]
# # 灵活的数据读取

#%% [markdown]
# ## Dataset

#%%
from torch.utils.data import Dataset


#%%
# 定义一个子类叫 custom_dataset，继承与 Dataset
class custom_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.transform = transform # 传入数据预处理
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        self.img_list = [i.split()[0] for i in lines] # 得到所有的图像名字
        self.label_list = [i.split()[1] for i in lines] # 得到所有的 label 

    def __getitem__(self, idx): # 根据 idx 取出其中一个
        img = self.img_list[idx]
        label = self.label_list[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self): # 总数据的多少
        return len(self.label_list)


#%%
txt_dataset = custom_dataset('./example_data/train.txt') # 读入 txt 文件


#%%
# 取得其中一个数据
data, label = txt_dataset[0]
print(data)
print(label)


#%%
# 再取一个
data2, label2 = txt_dataset[34]
print(data2)
print(label2)

#%% [markdown]
# 所以通过这种方式我们也能够非常方便的定义一个数据读入，同时也能够方便的定义数据预处理
#%% [markdown]
# ## DataLoader

#%%
from torch.utils.data import DataLoader


#%%
train_data1 = DataLoader(folder_set, batch_size=2, shuffle=True) # 将 2 个数据作为一个 batch


#%%
for im, label in train_data1: # 访问迭代器
    print(label)

#%% [markdown]
# 可以看到，通过训练我们可以访问到所有的数据，这些数据被分为了 5 个 batch，前面 4 个都有两个数据，最后一个 batch 只有一个数据，因为一共有 9 个数据，同时顺序也被打乱了
#%% [markdown]
# 下面我们用自定义的数据读入举例子

#%%
train_data2 = DataLoader(txt_dataset, 8, True) # batch size 设置为 8


#%%
im, label = next(iter(train_data2)) # 使用这种方式访问迭代器中第一个 batch 的数据


#%%
im


#%%
label

#%% [markdown]
# 现在有一个需求，希望能够将上面一个 batch 输出的 label 补成相同的长度，短的 label 用 0 填充，我们就需要使用 `collate_fn` 来自定义我们 batch 的处理方式，下面直接举例子

#%%
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True) # 将数据集按照 label 的长度从大到小排序
    img, label = zip(*batch) # 将数据和 label 配对取出
    # 填充
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = label[i]
        temp_label += '0' * (max_len - len(label[i]))
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    pad_label 
    return img, pad_label, lens # 输出 label 的真实长度

#%% [markdown]
# 使用我们自己定义 collate_fn 看看效果

#%%
train_data3 = DataLoader(txt_dataset, 8, True, collate_fn=collate_fn) # batch size 设置为 8


#%%
im, label, lens = next(iter(train_data3))


#%%
im


#%%
label


#%%
lens

#%% [markdown]
# 可以看到一个 batch 中所有的 label 都从长到短进行排列，同时短的 label 都被补长了，所以使用 collate_fn 能够非常方便的处理一个 batch 中的数据，一般情况下，没有特别的要求，使用 pytorch 中内置的 collate_fn 就可以满足要求了

