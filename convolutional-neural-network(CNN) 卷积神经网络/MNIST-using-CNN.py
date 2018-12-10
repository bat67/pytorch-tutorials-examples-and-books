#%% [markdown]
# # 卷积神经网络

#%%
import torch
import torch.utils.data
import torch.nn
import torch.optim
import torchvision.datasets
import torchvision.transforms

# 数据读取
train_dataset = torchvision.datasets.MNIST(root='./data/mnist',
        train=True, transform=torchvision.transforms.ToTensor(),
        download=True)
test_dataset = torchvision.datasets.MNIST(root='./data/mnist',
        train=False, transform=torchvision.transforms.ToTensor(),
        download=True)

batch_size = 100
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size)

# 搭建网络结构
class Net(torch.nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(
                torch.nn.Linear(128 * 14 * 14, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(1024, 10))
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 128 * 14 * 14)
        x = self.dense(x)
        return x

net = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters()) 

# 训练
num_epochs = 5
for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        preds = net(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        if idx % 100 == 0:
            print('epoch {}, batch {}, loss = {:g}'.format(
                    epoch, idx, loss.item()))

# 测试
correct = 0
total = 0
for images, labels in test_loader:
    preds = net(images)
    predicted = torch.argmax(preds, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
accuracy = correct / total
print('test acc: {:.1%}'.format(accuracy))


