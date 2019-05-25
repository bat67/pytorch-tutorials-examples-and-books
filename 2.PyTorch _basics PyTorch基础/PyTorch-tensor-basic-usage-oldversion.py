#%% [markdown]
# # PyTorch Tensor Basic Usage
# 
# - Create Tensor
# - Indexing,Joining,Slicing
# - Initialization
# - Math Operations
#%% [markdown]
# ## 1. Create Tensor
# ### 1) random numbers

#%%
import torch

# torch.rand(sizes) -> [0,1)
x = torch.rand(2,3)
x


#%%
# torch.randn(sizes) -> Z(0,1)
x = torch.randn(2,3)
x


#%%
# torch.randperm(n) -> permutation of 0~n
x = torch.randperm(5)
x

#%% [markdown]
# ### 2) zeros, ones, arange

#%%
# torch.zeros(2,3) -> [[0,0,0],[0,0,0]]
x = torch.zeros(2,3)
x


#%%
# torch.ones(2,3) -> [[0,0,0],[0,0,0]]
x = torch.ones(2,3)
x


#%%
# torch.arange(start,end,step=1) -> [start,end) with step
x = torch.arange(0,3,step=0.5)
x

#%% [markdown]
# ### 3) Tensor Data Type

#%%
# torch.FloatTensor(size or list)
x = torch.FloatTensor(2,3)
x


#%%
# torch.FloatTensor(size or list)
x = torch.FloatTensor([2,3])
x


#%%
# tensor.type_as(tensor_type)
x = x.type_as(torch.IntTensor())
x

#%% [markdown]
# ### 4) Numpy to Tensor, Tensor to Numpy

#%%
import numpy as np

# torch.from_numpy(ndarray) -> tensor

x1 = np.ndarray(shape=(2,3), dtype=int,buffer=np.array([1,2,3,4,5,6]))
x2 = torch.from_numpy(x1)

x2


#%%
# tensor.numpy() -> ndarray
x3 = x2.numpy()
x3

#%% [markdown]
# ### 5) Tensor on CPU & GPU

#%%
x = torch.FloatTensor([[1,2,3],[4,5,6]])
x


#%%
x_gpu = x.cuda()
x_gpu


#%%
x_cpu = x_gpu.cpu()
x_cpu

#%% [markdown]
# ### 6) Tensor Size

#%%
# tensor.size() -> indexing also possible

x = torch.FloatTensor(10,12,3,3)

x.size()[:]

#%% [markdown]
# ## 2. Indexing, Slicing, Joining, Reshaping
# ### 1) Indexing

#%%
# torch.index_select(input, dim, index)

x = torch.rand(4,3)
out = torch.index_select(x,0,torch.LongTensor([0,3]))

x,out


#%%
# pythonic indexing also works

x[:,0],x[0,:],x[0:2,0:2]


#%%
# torch.masked_select(input, mask)

x = torch.randn(2,3)
mask = torch.ByteTensor([[0,0,1],[0,1,0]])
out = torch.masked_select(x,mask)

x, mask, out

#%% [markdown]
# ### 2) Joining

#%%
# torch.cat(seq, dim=0) -> concatenate tensor along dim

x = torch.FloatTensor([[1,2,3],[4,5,6]])
y = torch.FloatTensor([[-1,-2,-3],[-4,-5,-6]])
z1 = torch.cat([x,y],dim=0)
z2 = torch.cat([x,y],dim=1)

x,y,z1,z2


#%%
# torch.stack(sequence,dim=0) -> stack along new dim

x = torch.FloatTensor([[1,2,3],[4,5,6]])
x_stack = torch.stack([x,x,x,x],dim=0)

x_stack

#%% [markdown]
# ### 3) Slicing

#%%
# torch.chunk(tensor, chunks, dim=0) -> tensor into num chunks

x_1, x_2 = torch.chunk(z1,2,dim=0)
y_1, y_2, y_3 = torch.chunk(z1,3,dim=1)

z1,x_1,x_2,z1,y_1,y_2,y_3


#%%
# torch.split(tensor,split_size,dim=0) -> split into specific size

x1,x2 = torch.split(z1,2,dim=0)
y1 = torch.split(z1,2,dim=1) 

z1,x1,x2,y1

#%% [markdown]
# ### 4) squeezing

#%%
# torch.squeeze(input,dim=None) -> reduce dim by 1

x1 = torch.FloatTensor(10,1,3,1,4)
x2 = torch.squeeze(x1)

x1.size(),x2.size()


#%%
# torch.unsqueeze(input,dim=None) -> add dim by 1

x1 = torch.FloatTensor(10,3,4)
x2 = torch.unsqueeze(x1,dim=0)

x1.size(),x2.size()

#%% [markdown]
# ### 5) Reshaping

#%%
# tensor.view(size)

x1 = torch.FloatTensor(10,3,4)
x2 = x1.view(-1)
x3 = x1.view(5,-1)
x4 = x1.view(3,10,-1)

x1.size(), x2.size(), x3.size(), x4.size()

#%% [markdown]
# ## 3. Initialization

#%%
import torch.nn.init as init

x1 = init.uniform(torch.FloatTensor(3,4),a=0,b=9) 
x2 = init.normal(torch.FloatTensor(3,4),std=0.2)
x3 = init.constant(torch.FloatTensor(3,4),3.1415)

x1,x2,x3

#%% [markdown]
# ## 4. Math Operations
# ### 1) Arithmetic operations

#%%
# torch.add()

x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
x2 = torch.FloatTensor([[1,2,3],[4,5,6]])
add = torch.add(x1,x2)

x1,x2,add,x1+x2,x1-x2


#%%
# torch.add() broadcasting

x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
x2 = torch.add(x1,10)

x1,x2,x1+10,x2-10


#%%
# torch.mul() -> size better match

x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
x2 = torch.FloatTensor([[1,2,3],[4,5,6]])
x3 = torch.mul(x1,x2)

x3


#%%
# torch.mul() -> broadcasting

x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
x2 = x1*10

x2


#%%
# torch.div() -> size better match

x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
x2 = torch.FloatTensor([[1,2,3],[4,5,6]])
x3 = torch.div(x1,x2)

x3


#%%
# torch.div() -> broadcasting

x1 = torch.FloatTensor([[1,2,3],[4,5,6]])

x1/5

#%% [markdown]
# ### 2) Other Math Operations

#%%
# torch.pow(input,exponent)

x1 = torch.FloatTensor(3,4)
torch.pow(x1,2),x1**2


#%%
# torch.exp(tensor,out=None) 

x1 = torch.FloatTensor(3,4)
torch.exp(x1)


#%%
# torch.log(input, out=None) -> natural logarithm

x1 = torch.FloatTensor(3,4)
torch.log(x1)

#%% [markdown]
# ### 3) Matrix operations

#%%
# torch.mm(mat1, mat2) -> matrix multiplication

x1 = torch.FloatTensor(3,4)
x2 = torch.FloatTensor(4,5)

torch.mm(x1,x2)


#%%
# torch.bmm(batch1, batch2) -> batch matrix multiplication

x1 = torch.FloatTensor(10,3,4)
x2 = torch.FloatTensor(10,4,5)

torch.bmm(x1,x2).size()


#%%
# torch.dot(tensor1,tensor2) -> dot product of two tensor

x1 = torch.FloatTensor(3,4)
x2 = torch.FloatTensor(3,4)

torch.dot(x1,x2)


#%%
# torch.t(matrix) -> transposed matrix

x1 = torch.FloatTensor(3,4)

x1,x1.t()


#%%
# torch.transpose(input,dim0,dim1) -> transposed matrix

x1 = torch.FloatTensor(10,3,4)

x1.size(), torch.transpose(x1,1,2).size(), x1.transpose(1,2).size()


#%%
# torch.eig(a,eigenvectors=False) -> eigen_value, eigen_vector

x1 = torch.FloatTensor(4,4)

x1,torch.eig(x1,True)


