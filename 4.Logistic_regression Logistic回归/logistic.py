import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

fr=open("data.txt")
lines=fr.readlines()

lines=[x.split('\n')[0] for x in lines]
#print(lines)
lines=[x.split(',') for x in lines]
x_data=np.array([[float(i[0]),float(i[1])] for i in lines],dtype=np.float32)
y_data=np.array([[float(i[2])] for i in lines ],dtype=np.float32)
x_data=torch.from_numpy(x_data)
y_data=torch.from_numpy(y_data)

#定义Logistic 回归模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.lr=nn.Linear(2,1)
        self.sm=nn.Sigmoid()

    def forward(self,x):
        x=self.lr(x)
        x=self.sm(x)
        return x
logistic_model=LogisticRegression()
#BCE 是二分类的损失函数
criterion=nn.BCELoss()
optimizer=optim.SGD(logistic_model.parameters(),lr=1e-3,momentum=0.9)

for epoch in range(5000):
    x = x_data
    y = y_data

    #前向
    out=logistic_model(x)
    loss=criterion(out,y)
    print_loss=loss.item()

    print("item:",loss.item())
    mask=out.ge(0.5).float() #输出大于0.5 就是1 ，小于0.5 就是0

    correct=(mask==y).sum()
    # if epoch + 1 == 1000:
    #     print(correct)
    acc=correct.item()/x.size(0)

    #后向
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch+1)%1000==0:
        print("*"*10)
        print("epoch:{}".format(epoch+1))
        print("loss is {:.4f}".format(print_loss))
        print("acc is{:.4f}".format(acc))
