#%% [markdown]
# # 线性模型和梯度下降
# 这是神经网络的第一课，我们会学习一个非常简单的模型，线性回归，同时也会学习一个优化算法-梯度下降法，对这个模型进行优化。线性回归是监督学习里面一个非常简单的模型，同时梯度下降也是深度学习中应用最广的优化算法，我们将从这里开始我们的深度学习之旅
#%% [markdown]
# ## 一元线性回归
# 一元线性模型非常简单，假设我们有变量 $x_i$ 和目标 $y_i$，每个 i 对应于一个数据点，希望建立一个模型
# 
# $$
# \hat{y}_i = w x_i + b
# $$
# 
# $\hat{y}_i$ 是我们预测的结果，希望通过 $\hat{y}_i$ 来拟合目标 $y_i$，通俗来讲就是找到这个函数拟合 $y_i$ 使得误差最小，即最小化
# 
# $$
# \frac{1}{n} \sum_{i=1}^n(\hat{y}_i - y_i)^2
# $$
#%% [markdown]
# 那么如何最小化这个误差呢？
# 
# 这里需要用到**梯度下降**，这是我们接触到的第一个优化算法，非常简单，但是却非常强大，在深度学习中被大量使用，所以让我们从简单的例子出发了解梯度下降法的原理
#%% [markdown]
# ## 梯度下降法
# 在梯度下降法中，我们首先要明确梯度的概念，随后我们再了解如何使用梯度进行下降。
#%% [markdown]
# ### 梯度
# 梯度在数学上就是导数，如果是一个多元函数，那么梯度就是偏导数。比如一个函数f(x, y)，那么 f 的梯度就是 
# 
# $$
# (\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y})
# $$
# 
# 可以称为 grad f(x, y) 或者 $\nabla f(x, y)$。具体某一点 $(x_0,\ y_0)$ 的梯度就是 $\nabla f(x_0,\ y_0)$。
# 
# 下面这个图片是 $f(x) = x^2$ 这个函数在 x=1 处的梯度
# 
# ![](https://ws3.sinaimg.cn/large/006tNc79ly1fmarbuh2j3j30ba0b80sy.jpg)
#%% [markdown]
# 梯度有什么意义呢？从几何意义来讲，一个点的梯度值是这个函数变化最快的地方，具体来说，对于函数 f(x, y)，在点 $(x_0, y_0)$ 处，沿着梯度 $\nabla f(x_0,\ y_0)$ 的方向，函数增加最快，也就是说沿着梯度的方向，我们能够更快地找到函数的极大值点，或者反过来沿着梯度的反方向，我们能够更快地找到函数的最小值点。
#%% [markdown]
# ### 梯度下降法
# 有了对梯度的理解，我们就能了解梯度下降发的原理了。上面我们需要最小化这个误差，也就是需要找到这个误差的最小值点，那么沿着梯度的反方向我们就能够找到这个最小值点。
# 
# 我们可以来看一个直观的解释。比如我们在一座大山上的某处位置，由于我们不知道怎么下山，于是决定走一步算一步，也就是在每走到一个位置的时候，求解当前位置的梯度，沿着梯度的负方向，也就是当前最陡峭的位置向下走一步，然后继续求解当前位置梯度，向这一步所在位置沿着最陡峭最易下山的位置走一步。这样一步步的走下去，一直走到觉得我们已经到了山脚。当然这样走下去，有可能我们不能走到山脚，而是到了某一个局部的山峰低处。
# 
# 类比我们的问题，就是沿着梯度的反方向，我们不断改变 w 和 b 的值，最终找到一组最好的 w 和 b 使得误差最小。
# 
# 在更新的时候，我们需要决定每次更新的幅度，比如在下山的例子中，我们需要每次往下走的那一步的长度，这个长度称为学习率，用 $\eta$ 表示，这个学习率非常重要，不同的学习率都会导致不同的结果，学习率太小会导致下降非常缓慢，学习率太大又会导致跳动非常明显，可以看看下面的例子
# 
# ![](https://ws2.sinaimg.cn/large/006tNc79ly1fmgn23lnzjg30980gogso.gif)
# 
# 可以看到上面的学习率较为合适，而下面的学习率太大，就会导致不断跳动
# 
# 最后我们的更新公式就是
# 
# $$
# w := w - \eta \frac{\partial f(w,\ b)}{\partial w} \\
# b := b - \eta \frac{\partial f(w,\ b)}{\partial b}
# $$
# 
# 通过不断地迭代更新，最终我们能够找到一组最优的 w 和 b，这就是梯度下降法的原理。
# 
# 最后可以通过这张图形象地说明一下这个方法
# 
# ![](https://ws3.sinaimg.cn/large/006tNc79ly1fmarxsltfqj30gx091gn4.jpg)
#%% [markdown]
# 上面是原理部分，下面通过一个例子来进一步学习线性模型

#%%
import torch
import numpy as np

torch.manual_seed(2017)


#%%
# 读入数据 x 和 y
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


#%%
# 画出图像
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(x_train, y_train, 'bo')


#%%
# 转换成 Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 定义参数 w 和 b
w = torch.randn(1, requires_grad=True) # 随机初始化
b = torch.zeros(1, requires_grad=True) # 使用 0 进行初始化


#%%
# 构建线性回归模型

def linear_model(x):
    return x * w + b


#%%
y_ = linear_model(x_train)

#%% [markdown]
# 经过上面的步骤我们就定义好了模型，在进行参数更新之前，我们可以先看看模型的输出结果长什么样

#%%
plt.plot(x_train.detach().numpy(), y_train.detach().numpy(), 'bo', label='real')
plt.plot(x_train.detach().numpy(), y_.detach().numpy(), 'ro', label='estimated')
plt.legend()

#%% [markdown]
# **思考：红色的点表示预测值，似乎排列成一条直线，请思考一下这些点是否在一条直线上？**
#%% [markdown]
# 这个时候需要计算我们的误差函数，也就是
# 
# $$
# \frac{1}{n} \sum_{i=1}^n(\hat{y}_i - y_i)^2
# $$

#%%
# 计算误差
def get_loss(y_, y):
    return torch.mean((y_ - y_train) ** 2)

loss = get_loss(y_, y_train)


#%%
# 打印一下看看 loss 的大小
print(loss)

#%% [markdown]
# 定义好了误差函数，接下来我们需要计算 w 和 b 的梯度了，这时得益于 PyTorch 的自动求导，我们不需要手动去算梯度，有兴趣的同学可以手动计算一下，w 和 b 的梯度分别是
# 
# $$
# \frac{\partial}{\partial w} = \frac{2}{n} \sum_{i=1}^n x_i(w x_i + b - y_i) \\
# \frac{\partial}{\partial b} = \frac{2}{n} \sum_{i=1}^n (w x_i + b - y_i)
# $$

#%%
# 自动求导
loss.backward()


#%%
# 查看 w 和 b 的梯度
print(w.grad)
print(b.grad)


#%%
# 更新一次参数

with torch.no_grad():
    w -= 1e-2 * w.grad
    b -= 1e-2 * b.grad

#%% [markdown]
# 更新完成参数之后，我们再一次看看模型输出的结果

#%%
y_ = linear_model(x_train)
plt.plot(x_train.detach().numpy(), y_train.detach().numpy(), 'bo', label='real')
plt.plot(x_train.detach().numpy(), y_.detach().numpy(), 'ro', label='estimated')
plt.legend()

#%% [markdown]
# 从上面的例子可以看到，更新之后红色的线跑到了蓝色的线下面，没有特别好的拟合蓝色的真实值，所以我们需要在进行几次更新

#%%
for e in range(10): # 进行 10 次更新
    y_ = linear_model(x_train)
    loss = get_loss(y_, y_train)
    
    w.grad.zero_() # 记得归零梯度
    b.grad.zero_() # 记得归零梯度
    loss.backward()
    
    with torch.no_grad():
        w -= 1e-2 * w.grad
        b -= 1e-2 * b.grad
    
    print('epoch: {}, loss: {}'.format(e, loss.item()))


#%%
y_ = linear_model(x_train)
plt.plot(x_train.detach().numpy(), y_train.detach().numpy(), 'bo', label='real')
plt.plot(x_train.detach().numpy(), y_.detach().numpy(), 'ro', label='estimated')
plt.legend()

#%% [markdown]
# 经过 10 次更新，我们发现红色的预测结果已经比较好的拟合了蓝色的真实值。
# 
# 现在你已经学会了你的第一个机器学习模型了，再接再厉，完成下面的小练习。
#%% [markdown]
# **小练习：**
# 
# 重启 notebook 运行上面的线性回归模型，但是改变训练次数以及不同的学习率进行尝试得到不同的结果
#%% [markdown]
# ## 多项式回归模型
#%% [markdown]
# 下面我们更进一步，讲一讲多项式回归。什么是多项式回归呢？非常简单，根据上面的线性回归模型
# 
# $$
# \hat{y} = w x + b
# $$
# 
# 这里是关于 x 的一个一次多项式，这个模型比较简单，没有办法拟合比较复杂的模型，所以我们可以使用更高次的模型，比如
# 
# $$
# \hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \cdots
# $$
# 
# 这样就能够拟合更加复杂的模型，这就是多项式模型，这里使用了 x 的更高次，同理还有多元回归模型，形式也是一样的，只是出了使用 x，还是更多的变量，比如 y、z 等等，同时他们的 loss 函数和简单的线性回归模型是一致的。
#%% [markdown]
# 首先我们可以先定义一个需要拟合的目标函数，这个函数是个三次的多项式

#%%
# 定义一个多变量函数

w_target = np.array([0.5, 3, 2.4]) # 定义参数
b_target = np.array([0.9]) # 定义参数

f_des = 'y = {:.2f} + {:.2f} * x + {:.2f} * x^2 + {:.2f} * x^3'.format(
    b_target[0], w_target[0], w_target[1], w_target[2]) # 打印出函数的式子

print(f_des)

#%% [markdown]
# 我们可以先画出这个多项式的图像

#%%
# 画出这个函数的曲线
x_sample = np.arange(-3, 3.1, 0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3

plt.plot(x_sample, y_sample, label='real curve')
plt.legend()

#%% [markdown]
# 接着我们可以构建数据集，需要 x 和 y，同时是一个三次多项式，所以我们取了 $x,\ x^2, x^3$

#%%
# 构建数据 x 和 y
# x 是一个如下矩阵 [x, x^2, x^3]
# y 是函数的结果 [y]

x_train = np.stack([x_sample ** i for i in range(1, 4)], axis=1)
x_train = torch.from_numpy(x_train).float() # 转换成 float tensor

y_train = torch.from_numpy(y_sample).float().unsqueeze(1) # 转化成 float tensor 

#%% [markdown]
# 接着我们可以定义需要优化的参数，就是前面这个函数里面的 $w_i$

#%%
# 定义参数和模型
w = torch.randn((3, 1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

def multi_linear(x):
    return torch.mm(x, w) + b

#%% [markdown]
# 我们可以画出没有更新之前的模型和真实的模型之间的对比

#%%
# 画出更新之前的模型
y_pred = multi_linear(x_train)

plt.plot(x_train.detach().numpy()[:, 0], y_pred.detach().numpy(), label='fitting curve', color='r')
plt.plot(x_train.detach().numpy()[:, 0], y_sample, label='real curve', color='b')
plt.legend()

#%% [markdown]
# 可以发现，这两条曲线之间存在差异，我们计算一下他们之间的误差

#%%
# 计算误差，这里的误差和一元的线性模型的误差是相同的，前面已经定义过了 get_loss
loss = get_loss(y_pred, y_train)
print(loss)


#%%
# 自动求导
loss.backward()


#%%
# 查看一下 w 和 b 的梯度
print(w.grad)
print(b.grad)


#%%
# 更新一下参数
with torch.no_grad():
    w -= 0.001 * w.grad
    b -= 0.001 * b.grad


#%%
# 画出更新一次之后的模型
y_pred = multi_linear(x_train)

plt.plot(x_train.detach().numpy()[:, 0], y_pred.detach().numpy(), label='fitting curve', color='r')
plt.plot(x_train.detach().numpy()[:, 0], y_sample, label='real curve', color='b')
plt.legend()

#%% [markdown]
# 因为只更新了一次，所以两条曲线之间的差异仍然存在，我们进行 100 次迭代

#%%
# 进行 100 次参数更新
for e in range(100):
    y_pred = multi_linear(x_train)
    loss = get_loss(y_pred, y_train)
    
    w.grad.zero_()
    b.grad.zero_()
    loss.backward()
    # 更新参数
    with torch.no_grad():
        w -= 0.001 * w.grad
        b -= 0.001 * b.grad  

    if (e + 1) % 20 == 0:
        print('epoch {}, Loss: {:.5f}'.format(e+1, loss.item()))

#%% [markdown]
# 可以看到更新完成之后 loss 已经非常小了，我们画出更新之后的曲线对比

#%%
# 画出更新之后的结果
y_pred = multi_linear(x_train)

plt.plot(x_train.detach().numpy()[:, 0], y_pred.detach().numpy(), label='fitting curve', color='r')
plt.plot(x_train.detach().numpy()[:, 0], y_sample, label='real curve', color='b')
plt.legend()

#%% [markdown]
# 可以看到，经过 100 次更新之后，可以看到拟合的线和真实的线已经完全重合了
#%% [markdown]
# **小练习：上面的例子是一个三次的多项式，尝试使用二次的多项式去拟合它，看看最后能做到多好**
# 
# **提示：参数 `w = torch.randn(2, 1)`，同时重新构建 x 数据集**

