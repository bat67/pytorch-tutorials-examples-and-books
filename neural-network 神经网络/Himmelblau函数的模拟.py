#%% [markdown]
# # Himmelblau 函数的模拟

#%%
import torch
import torch.nn as nn
import torch.optim

#%% [markdown]
# 生成数据

#%%
torch.manual_seed(seed=0) # 固定随机数种子,这样生成的数据是确定的
sample_num = 1000 # 生成样本数
features = torch.rand(sample_num, 2)  * 12 - 6 # 特征数据
noises = torch.randn(sample_num)
def himmelblau(x):
    return (x[:,0] ** 2 + x[:,1] - 11) ** 2 + (x[:,0] + x[:,1] ** 2 - 7) ** 2
hims = himmelblau(features) * 0.01
labels = hims + noises # 标签数据


#%%
train_num, validate_num, test_num = 600, 200, 200 # 分割数据
train_mse = torch.mean(noises[:train_num] ** 2)
validate_mse = torch.mean(noises[train_num:-test_num] ** 2)
test_mse = torch.mean(noises[-test_num:] ** 2)
print ('真实:训练集MSE = {:g}, 验证集MSE = {:g}, 测试集MSE = {:g}'.format(
        train_mse, validate_mse, test_mse))
# 输出: 真实:训练集MSE = 0.918333, 验证集MSE = 0.902182, 测试集MSE = 0.978382

#%% [markdown]
# 搭建神经网络

#%%
hidden_features = [6, 2] # 指定隐含层数
layers = [nn.Linear(2, hidden_features[0]),]
for idx, hidden_feature in enumerate(hidden_features):
    layers.append(nn.Sigmoid())
    next_hidden_feature = hidden_features[idx + 1]             if idx + 1 < len(hidden_features) else 1
    layers.append(nn.Linear(hidden_feature, next_hidden_feature))
net = nn.Sequential(*layers) # 前馈神经网络
print('神经网络为 {}'.format(net))

#%% [markdown]
# 训练网络

#%%
optimizer = torch.optim.Adam(net.parameters())
criterion = nn.MSELoss()

train_entry_num = 600 # 选择训练样本数
    
nIteration = 100000 # 00 # 最大迭代次数
for step in range(nIteration):
    outputs = net(features)
    preds = outputs[:, 0]
    
    loss_train = criterion(preds[:train_entry_num],
            labels[:train_entry_num])
    loss_validate = criterion(preds[train_num:-test_num],
            labels[train_num:-test_num])
    if step % 10000 == 0:
        print ('#{} 训练集MSE = {:g}, 验证集MSE = {:g}'.format(
                step, loss_train, loss_validate))

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

print ('训练集MSE = {:g}, 验证集MSE = {:g}'.format(loss_train, loss_validate))


