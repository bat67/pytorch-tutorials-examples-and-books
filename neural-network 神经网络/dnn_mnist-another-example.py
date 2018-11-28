#%%
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

#%% Hyper Parameters   配置参数
torch.manual_seed(1)  # 设置随机数种子，确保结果可重复
input_size = 784  #
hidden_size = 500
num_classes = 10
num_epochs = 20  # 训练次数
batch_size = 100  # 批处理大小
learning_rate = 0.001  # 学习率

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = r"C:\DATASETS\mnist"

#%% MNIST Dataset  下载训练集 MNIST 手写数字训练集
train_dataset = dsets.MNIST(root=root,  # 数据保持的位置
                            train=True,  # 训练集
                            transform=transforms.ToTensor(),  # 一个取值范围是[0,255]的PIL.Image
                            # 转化为取值范围是[0,1.0]的torch.FloadTensor
                            download=True)  # 下载数据

test_dataset = dsets.MNIST(root=root,
                           train=False,  # 测试集
                           transform=transforms.ToTensor())

#%% Data Loader (Input Pipeline)
# 数据的批处理，尺寸大小为batch_size,
# 在训练集中，shuffle 必须设置为True, 表示次序是随机的
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#%% Neural Network Model (1 hidden layer)  定义神经网络模型
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes).to(device)
#%% 打印模型
print(net)

#%% Loss and Optimizer  定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#%% Train the Model   开始训练
net.train()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  # 批处理
        # Convert torch tensor to Variable
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer  #梯度清零，以免影响其他batch
        outputs = net(images)  # 前向传播

        # import pdb
        # pdb.set_trace()
        loss = criterion(outputs, labels)  # loss 
        loss.backward()  # 后向传播，计算梯度
        optimizer.step()  # 梯度更新

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

#%% Test the Model
correct = 0
total = 0
net.eval()
with torch.no_grad():
    for images, labels in test_loader:  # test set 批处理
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.detach(), 1)  # 预测结果
        total += labels.size(0)  # 正确结果
        correct += (predicted == labels).sum()  # 正确结果总数

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#%% Save the Model
torch.save(net.state_dict(), 'mnist_dnn_model.pkl')


#%%
