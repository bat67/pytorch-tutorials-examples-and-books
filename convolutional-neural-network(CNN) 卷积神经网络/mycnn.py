# 配置库
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# 配置参数
torch.manual_seed(
    1)  # 设置随机数种子，确保结果可重复
batch_size = 128  # 批处理大小
learning_rate = 1e-2  # 学习率
num_epoches = 10  # 训练次数

# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root=r'C:/DATASETS/mnist/',  # 数据保持的位置
    train=True,  # 训练集
    transform=transforms.ToTensor(),  # 一个取值范围是[0,255]的PIL.Image
    # 转化为取值范围是[0,1.0]的torch.FloadTensor
    download=True)  # 下载数据

test_dataset = datasets.MNIST(
    root=r'C:/DATASETS/mnist/',
    train=False,  # 测试集
    transform=transforms.ToTensor())

# 数据的批处理，尺寸大小为batch_size,
# 在训练集中，shuffle 必须设置为True, 表示次序是随机的
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义卷积神经网络模型
class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):  # 28x28x1
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),  # 28 x28
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 14 x 14
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # 10 * 10*16
            nn.ReLU(True), nn.MaxPool2d(2, 2))  # 5x5x16

        self.fc = nn.Sequential(
            nn.Linear(400, 120),  # 400 = 5 * 5 * 16
            nn.Linear(120, 84),
            nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), 400)  # 400 = 5 * 5 * 16, 
        out = self.fc(out)
        return out




model = Cnn(1, 10)  # 图片大小是28x28, 10
# 打印模型
print(model)
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 开始训练
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):  # 批处理
        img, label = data

        # 前向传播 
        out = model(img)
        loss = criterion(out, label)  # loss
        running_loss += loss.item() * label.size(0)  # total loss , 由于loss 是batch 取均值的，需要把batch size 乘回去
        _, pred = torch.max(out, 1)  # 预测结果
        num_correct = (pred == label).sum()  # 正确结果的num
        # accuracy = (pred == label).float().mean() #正确率
        running_acc += num_correct.item()  # 正确结果的总数
        # 后向传播
        optimizer.zero_grad()  # 梯度清零，以免影响其他batch
        loss.backward()  # 后向传播，计算梯度
        optimizer.step()  # 梯度更新

        # if i % 300 == 0:
        #    print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
        #        epoch + 1, num_epoches, running_loss / (batch_size * i),
        #        running_acc / (batch_size * i)))
    # 打印一个循环后，训练集合上的loss 和 正确率
    print('Train Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))

# 模型测试， 由于训练和测试 BatchNorm, Dropout配置不同，需要说明是否模型测试
model.eval()
with torch.no_grad():
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:  # test set 批处理
        img, label = data

        out = model(img)  # 前向算法 
        loss = criterion(out, label)  # 计算 loss
        eval_loss += loss.item() * label.size(0)  # total loss
        _, pred = torch.max(out, 1)  # 预测结果
        num_correct = (pred == label).sum()  # 正确结果
        eval_acc += num_correct.item()  # 正确结果总数

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc * 1.0 / (len(test_dataset))))

# 保存模型
torch.save(model.state_dict(), './cnn.pth')
