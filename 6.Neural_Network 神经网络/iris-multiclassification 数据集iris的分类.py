'''
PyTorch 实现iris 数据集的分类。
'''
#%%
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from torch.optim import SGD
import matplotlib.pyplot as plt

#%%
# GPU 是否可用
use_cuda = torch.cuda.is_available()
print("use_cuda: ", use_cuda)

#%%
# 加载数据集
iris = load_iris()
print(iris.keys())  # dict_keys(['target_names', 'data', 'feature_names', 'DESCR', 'target'])

#%%
x = iris['data']  # 特征信息
y = iris['target']  # 目标分类
print(x.shape)  # (150, 4)
print(x.shape)  # (150,)

print(y)

#%%
x = torch.FloatTensor(x)
y = torch.LongTensor(y)

#%%
class Net(nn.Module):
    """
    定义网络
    """

    def __init__(self, n_feature, n_hidden, n_output):
        """
        初始化函数，接受自定义输入特征维数，隐藏层特征维数，输出层特征维数
        """
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 一个线性隐藏层
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 线性输出层

    def forward(self, x):
        """
        前向传播过程
        """
        x = torch.sigmoid(self.hidden(x))
        x = self.predict(x)
        out = F.log_softmax(x, dim=1)
        return out


#%% iris 中输入特征 4 维，隐藏层和输出层可以自己选择
net = Net(n_feature=4, n_hidden=5, n_output=4)

#%% 如果GPU可用 训练数据和模型都放到GPU上，注意：数据和网络是否在GPU上要同步
if use_cuda:
    x = x.cuda()
    y = y.cuda()
    net = net.cuda()

#%% 查看网络结构
print(net)
#%%
optimizer = SGD(net.parameters(), lr=0.5)
#%%
iter_num = 1000
px, py = [], []

plt.rcParams['font.sans-serif'] = ['STSong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#%%
for i in range(iter_num):
    # 数据集传入网络前向计算
    prediction = net(x)

    # 计算loss
    loss = F.nll_loss(prediction, y)
    # 这里也可用CrossEntropyLoss
    # loss = loss_func(prediction, y)

    # 清除网络状态
    optimizer.zero_grad()

    # loss 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 打印并记录当前的index 和 loss
    print(i, " loss: ", loss.item())
    px.append(i)
    py.append(loss.item())

    if i % 10 == 0:
        # 动态画出loss走向 结果：loss.png
        plt.cla()
        plt.title(u'训练过程的loss曲线')
        plt.xlabel(u'迭代次数')
        plt.ylabel('损失')
        plt.plot(px, py, 'r-', lw=1)
        plt.text(0, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    if i == iter_num - 1:
        # 最后一个图像定格
        plt.show()
