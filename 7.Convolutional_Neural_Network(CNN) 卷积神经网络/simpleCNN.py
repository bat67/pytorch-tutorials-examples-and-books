import torch
from torch import nn, optim

class SimpleCNN(nn.Module) :
    def __init__(self) :
        # b, 3, 32, 32
        super().__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv_1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1))
        # b, 32, 32, 32
        layer1.add_module('relu_1', nn.ReLU(True))
        layer1.add_module('pool_1', nn.MaxPool2d(2, 2))  # b, 32, 16, 16
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv_2', nn.Conv2d(32, 64, 3, 1, padding=1))
        # b, 64, 16, 16
        layer2.add_module('relu_2', nn.ReLU(True))
        layer2.add_module('pool_2', nn.MaxPool2d(2, 2))  # b, 64, 8, 8
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv_3', nn.Conv2d(64, 128, 3, 1, padding=1))
        # b, 128, 8, 8
        layer3.add_module('relu_3', nn.ReLU(True))
        layer3.add_module('pool_3', nn.MaxPool2d(2, 2))  # b, 128, 4, 4
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc_1', nn.Linear(2048, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc_2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc_3', nn.Linear(64, 10))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        fc_input = conv3.view(conv3.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out



model = SimpleCNN()
print(model)

a = torch.FloatTensor([[1,2,3],[4,5,6]])
print(a)

