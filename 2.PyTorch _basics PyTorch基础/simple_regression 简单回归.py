#%%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from itertools import count

#%%
random_state = 5000
torch.manual_seed(random_state)
POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5

#%%
def make_features(x):
    """创建一个特征矩阵结构为[x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], 1)


def f(x):
    """近似函数."""
    return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    """生成多向式描述内容."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """创建类似 (x, f(x))的批数据."""
    random = torch.from_numpy(np.sort(torch.randn(batch_size)))
    x = make_features(random)
    y = f(x)
    return x, y


#%% 声明模型
fc = torch.nn.Linear(W_target.size(0), 1)

for batch_idx in count(1):
    # 获取数据
    batch_x, batch_y = get_batch()

    # 重置求导
    fc.zero_grad()

    # 前向传播
    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.item()

    # 后向传播
    output.backward()

    # 应用导数
    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad.data)

    # 停止条件
    if loss < 1e-3:
        plt.cla()
        plt.scatter(batch_x.data.numpy()[:, 0], batch_y.data.numpy()[:, 0], label='real curve', color='b')
        plt.plot(batch_x.data.numpy()[:, 0], fc(batch_x).data.numpy()[:, 0], label='fitting curve', color='r')
        plt.legend()
        plt.show()
        break

#%%
print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
