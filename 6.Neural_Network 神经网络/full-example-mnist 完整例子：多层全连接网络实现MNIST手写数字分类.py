#%%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
class Simple_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Simple_Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2 ,out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#%%
class Activation_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


#%%
class Batch_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


#%%
batch_size = 64
learning_rate = 1e-2
num_epoches = 20

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化
    x = x.reshape(-1) # 拉平
    x = torch.from_numpy(x)
    return x


datasets_root = r"C:/DATASETS"

train_set = datasets.MNIST(root=datasets_root, train=True, transform=data_tf, download=True)
test_set = datasets.MNIST(root=datasets_root, train=False, transform=data_tf, download=True)

train_dataset = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataset = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#%%

D_in, H1, H2, D_out = 28, 300, 100, 10

model = Activation_Net(D_in*D_in, H1, H2, D_out).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#%%

epoch_num = 50

losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(epoch_num):
    train_loss = 0
    train_acc = 0
    model.train()
    for im, label in train_dataset:
        # 前向传播
        im = im.to(device)
        label = label.to(device)
        
        out = model(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(dim = 1)
#怎么就一定保证是按照下标的循序来的？因为loss = criterion(out, label)这句就是对应的关系，要是想不对应就是loss函数怎么设置了
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_dataset))
    acces.append(train_acc / len(train_dataset))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    model.eval() # 将模型改为预测模式
    with torch.no_grad():
        for im, label in test_dataset:
            
            im = im.to(device)
            label = label.to(device)
            
            out = model(im)
            loss = criterion(out, label)
            # 记录误差
            eval_loss += loss.item()
            # 记录准确率
            _, pred = out.max(1)

            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            eval_acc += acc
        
    eval_losses.append(eval_loss / len(test_dataset))
    eval_acces.append(eval_acc / len(test_dataset))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'.format( e, train_loss / len(train_dataset), train_acc / len(train_dataset), eval_loss / len(test_dataset), eval_acc / len(test_dataset)))


