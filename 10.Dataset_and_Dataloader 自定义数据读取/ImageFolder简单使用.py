#%%
from torchvision.datasets import ImageFolder


#%%
# 三个文件夹，每个文件夹一共有 3 张图片作为例子
folder_set = ImageFolder('./example_data/image/')


#%%
# 查看名称和类别下标的对应
folder_set.class_to_idx


#%%
# 得到所有的图片名字和标签
folder_set.imgs


#%%
# 取出其中一个数据
im, label = folder_set[0]


#%%
im


#%%
label


#%%
from torchvision import transforms as tfs


#%%
# 传入数据预处理方式
data_tf = tfs.ToTensor()

folder_set = ImageFolder('./example_data/image/', transform=data_tf)

im, label = folder_set[0]


#%%
im


#%%
label

#%% [markdown]
# 可以看到通过这种方式能够非常方便的访问每个数据点