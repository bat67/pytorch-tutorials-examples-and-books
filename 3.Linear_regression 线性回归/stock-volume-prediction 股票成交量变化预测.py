#%% [markdown]
# # 股票成交量变化预测

#%%
from pandas_datareader.data import DataReader

import torch
import torch.nn
import torch.optim

#%% [markdown]
# 数据读取

#%%
df = DataReader('FB.US', 'quandl', '2012-01-01', '2018-02-01')
df


#%%
train_start, train_end = sum(df.index >= '2017'), sum(df.index >= '2013')
test_start, test_end = sum(df.index >= '2018'), sum(df.index >= '2017')
n_total_train = train_end - train_start
n_total_test = test_end - test_start
s_mean = df[train_start:train_end].mean()
s_std = df[train_start:train_end].std()
n_features = 5
df_feature = ((df - s_mean) / s_std).iloc[:, :n_features]
s_label = (df['Volume'] < df['Volume'].shift(1)).astype(int)
df_feature, s_label

#%% [markdown]
# 训练和测试

#%%
fc = torch.nn.Linear(n_features, 1)
weights, bias = fc.parameters()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(fc.parameters())

x = torch.tensor(df_feature.values, dtype=torch.float32)
y = torch.tensor(s_label.values.reshape(-1, 1), dtype=torch.float32)

n_step = 20001
for step in range(n_step):
    if step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pred = fc(x)
    loss = criterion(pred[train_start:train_end], y[train_start:train_end])
    
    if step % 500 == 0:
        # print('#{}, 损失 = {:g}, 权重 weights = {}, bias = {:g}'.format(
        #         step, loss, weights[0, :].tolist(), bias.item()))
        print('#{}, 损失 = {:g}, '.format(step, loss))
        
        output = (pred > 0)
        correct = (output == y.byte())
        n_correct_train = correct[train_start:train_end].sum().item()
        n_correct_test = correct[test_start:test_end].sum().item()
        accuracy_train = n_correct_train / n_total_train
        accuracy_test = n_correct_test / n_total_test
        print('训练集准确率 = {}, 测试集准确率 = {}'.format(accuracy_train, accuracy_test))


