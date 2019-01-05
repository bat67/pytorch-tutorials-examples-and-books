#%% [markdown]
# # Introduction to Deep Learning with PyTorch
# 
# In this notebook, you'll get introduced to [PyTorch](http://pytorch.org/), a framework for building and training neural networks. PyTorch in a lot of ways behaves like the arrays you love from Numpy. These Numpy arrays, after all, are just tensors. PyTorch takes these tensors and makes it simple to move them to GPUs for the faster processing needed when training neural networks. It also provides a module that automatically calculates gradients (for backpropagation!) and another module specifically for building neural networks. All together, PyTorch ends up being more coherent with Python and the Numpy/Scipy stack compared to TensorFlow and other frameworks.
# 
# 
#%% [markdown]
# ## Neural Networks
# 
# Deep Learning is based on artificial neural networks which have been around in some form since the late 1950s. The networks are built from individual parts approximating neurons, typically called units or simply "neurons." Each unit has some number of weighted inputs. These weighted inputs are summed together (a linear combination) then passed through an activation function to get the unit's output.
# 
# <img src="assets/simple_neuron.png" width=400px>
# 
# Mathematically this looks like: 
# 
# $$
# \begin{align}
# y &= f(w_1 x_1 + w_2 x_2 + b) \\
# y &= f\left(\sum_i w_i x_i \right)
# \end{align}
# $$
# 
# With vectors this is the dot/inner product of two vectors:
# 
# $$
# h = \begin{bmatrix}
# x_1 \, x_2 \cdots  x_n
# \end{bmatrix}
# \cdot 
# \begin{bmatrix}
#            w_1 \\
#            w_2 \\
#            \vdots \\
#            w_n
# \end{bmatrix}
# $$
#%% [markdown]
# ### Stack them up!
# 
# We can assemble these unit neurons into layers and stacks, into a network of neurons. The output of one layer of neurons becomes the input for the next layer. With multiple input units and output units, we now need to express the weights as a matrix.
# 
# <img src='assets/multilayer_diagram_weights.png' width=450px>
# 
# We can express this mathematically with matrices again and use matrix multiplication to get linear combinations for each unit in one operation. For example, the hidden layer ($h_1$ and $h_2$ here) can be calculated 
# 
# $$
# \vec{h} = [h_1 \, h_2] = 
# \begin{bmatrix}
# x_1 \, x_2 \cdots \, x_n
# \end{bmatrix}
# \cdot 
# \begin{bmatrix}
#            w_{11} & w_{12} \\
#            w_{21} &w_{22} \\
#            \vdots &\vdots \\
#            w_{n1} &w_{n2}
# \end{bmatrix}
# $$
# 
# The output for this small network is found by treating the hidden layer as inputs for the output unit. The network output is expressed simply
# 
# $$
# y =  f_2 \! \left(\, f_1 \! \left(\vec{x} \, \mathbf{W_1}\right) \mathbf{W_2} \right)
# $$
#%% [markdown]
# ## Tensors
# 
# It turns out neural network computations are just a bunch of linear algebra operations on *tensors*, a generalization of matrices. A vector is a 1-dimensional tensor, a matrix is a 2-dimensional tensor, an array with three indices is a 3-dimensional tensor (RGB color images for example). The fundamental data structure for neural networks are tensors and PyTorch (as well as pretty much every other deep learning framework) is built around tensors.
# 
# <img src="assets/tensor_examples.svg" width=600px>
# 
# With the basics covered, it's time to explore how we can use PyTorch to build a simple neural network.

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import torch

import helper

#%% [markdown]
# First, let's see how we work with PyTorch tensors. These are the fundamental data structures of neural networks and PyTorch, so it's imporatant to understand how these work.

#%%
x = torch.rand(3, 2)
x


#%%
y = torch.ones(x.size())
y


#%%
z = x + y
z

#%% [markdown]
# In general PyTorch tensors behave similar to Numpy arrays. They are zero indexed and support slicing.

#%%
z[0]


#%%
z[:, 1:]

#%% [markdown]
# Tensors typically have two forms of methods, one method that returns another tensor and another method that performs the operation in place. That is, the values in memory for that tensor are changed without creating a new tensor. In-place functions are always followed by an underscore, for example `z.add()` and `z.add_()`.

#%%
# Return a new tensor z + 1
z.add(1)


#%%
# z tensor is unchanged
z


#%%
# Add 1 and update z tensor in-place
z.add_(1)


#%%
# z has been updated
z

#%% [markdown]
# ### Reshaping
# 
# Reshaping tensors is a really common operation. First to get the size and shape of a tensor use `.size()`. Then, to reshape a tensor, use `.resize_()`. Notice the underscore, reshaping is an in-place operation.

#%%
z.size()


#%%
z.resize_(2, 3)


#%%
z

#%% [markdown]
# ## Numpy to Torch and back
# 
# Converting between Numpy arrays and Torch tensors is super simple and useful. To create a tensor from a Numpy array, use `torch.from_numpy()`. To convert a tensor to a Numpy array, use the `.numpy()` method.

#%%
a = np.random.rand(4,3)
a


#%%
b = torch.from_numpy(a)
b


#%%
b.numpy()

#%% [markdown]
# The memory is shared between the Numpy array and Torch tensor, so if you change the values in-place of one object, the other will change as well.

#%%
# Multiply PyTorch Tensor by 2, in place
b.mul_(2)


#%%
# Numpy array matches new values from Tensor
a


