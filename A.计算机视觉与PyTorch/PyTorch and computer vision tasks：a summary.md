- [PyTorch与计算机视觉简要总结](#pytorch与计算机视觉简要总结)
  - [1. 使用预训练好的 Resnet 网络进行微调](#1-使用预训练好的-resnet-网络进行微调)
    - [重要提示](#重要提示)
  - [2. 模型的保存和加载](#2-模型的保存和加载)
  - [3. 修改、删除或增加最后一层](#3-修改删除或增加最后一层)
    - [修改最后一层](#修改最后一层)
    - [删除最后一层 (通常，在需要一个层的参数时)](#删除最后一层-通常在需要一个层的参数时)
    - [增加层](#增加层)
  - [4. 自定义模型 : 结合 Section 1-3，在模型头部添加层](#4-自定义模型--结合-section-1-3在模型头部添加层)
  - [5. 自定义损失函数](#5-自定义损失函数)


# PyTorch与计算机视觉简要总结

```python
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
```

```python
model = models.resnet18(pretrained=False)
```

## 1. 使用预训练好的 Resnet 网络进行微调

让我们首先研究这个模型的各个层，然后再决定要固定（freeze）哪些层。固定的意思是我们想要这些层的参数固定不变。微调简单来说就是使用一个在大规模数据集上预训练好模型在我们的目标数据集上接着训练。当然，我们也可以不微调从零开始训练，这意味的重新造轮子，后面会解释为什么。

假设，我想训练一个数据集来学习区分汽车和自行车。现在，我可以收集这两个类别的图像，并从头开始训练网络。但是，考虑到现有的大部分工作，很容易找到一个训练有素的模型来识别狗、猫和人。无可否认，这三种汽车看起来既不像汽车也不像自行车。然而，这总比什么都没有好。我们可以从这个模型开始，训练它学习区分汽车和自行车。

好处有:

* 它会更快，
* 我们需要更少的猫和自行车的图像。

如果对迁移学习感兴趣的话，可以参考：http://cs231n.github.io/transfer-learning

现在，让我们来看看resnet18的内容。为此，我们使用`.children()`函数。这让我们看看模型不同层的内容。然后，我们使用`.parameters()`函数访问任意层的参数/权重。最后，每个参数都有一个属性`.requires_grad`，它定义了一个参数是训练的还是冻结的。默认情况下，它是`True`，网络在每次迭代中都会更新它。如果将其设置为False，则不更新它，并称为“冻结”。


```python
child_counter = 0
for child in model.children():
    print(" child", child_counter, "is -")
    print(child)
    child_counter += 1
```

输出：
```python
child 0 is -
Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
 child 1 is -
BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
 child 2 is -
ReLU(inplace)
 child 3 is -
MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
 child 4 is -
Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (1): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
 child 5 is -
Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (downsample): Sequential(
      (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (1): BasicBlock(
    (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
 child 6 is -
Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (downsample): Sequential(
      (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (1): BasicBlock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
 child 7 is -
Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (downsample): Sequential(
      (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (1): BasicBlock(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
 child 8 is -
AvgPool2d(kernel_size=7, stride=1, padding=0)
 child 9 is -
Linear(in_features=512, out_features=1000, bias=True)
```

现在，您可以看到一些子元素实际上是一大块一大块的，它们内部有层。要更深入地访问一层，我们还可以在一个子对象上运行`.children()`

让我们看看我们想要冻结所有参数直到Child 6的第一个`BasicBlock`。首先，让我们查看一个参数并将其设置为`frozen`


```python
for child in model.children():
    for param in child.parameters():
        print("This is what a parameter looks like - \n",param)
        break
    break
```

输出：
```python
This is what a parameter looks like - 
 Parameter containing:
tensor([[[[ 0.0215,  0.0528,  0.0342,  ...,  0.0086, -0.0114,  0.0041],
          [ 0.0004,  0.0073,  0.0066,  ..., -0.0115, -0.0238, -0.0034],
          [-0.0017, -0.0078, -0.0114,  ...,  0.0236,  0.0338,  0.0080],
          ...,
          [ 0.0261,  0.0273, -0.0231,  ...,  0.0469, -0.0048,  0.0094],
          [-0.0027,  0.0285,  0.0030,  ..., -0.0260,  0.0206,  0.0365],
          [-0.0185,  0.0175, -0.0042,  ..., -0.0078, -0.0132, -0.0199]],

         [[ 0.0152, -0.0347,  0.0170,  ...,  0.0012,  0.0502, -0.0005],
          [-0.0185, -0.0213, -0.0167,  ...,  0.0144, -0.0169,  0.0038],
          [-0.0605, -0.0036, -0.0019,  ..., -0.0112,  0.0087,  0.0223],
          ...,
          [-0.0331, -0.0077,  0.0107,  ..., -0.0040, -0.0057,  0.0158],
          [-0.0075,  0.0082, -0.0406,  ...,  0.0102, -0.0123, -0.0018],
          [-0.0211,  0.0258, -0.0119,  ..., -0.0261,  0.0303,  0.0390]],

         [[-0.0280,  0.0115, -0.0209,  ...,  0.0384, -0.0050, -0.0155],
          [-0.0511,  0.0144, -0.0240,  ..., -0.0469, -0.0738, -0.0017],
          [-0.0222, -0.0044, -0.0265,  ..., -0.0671, -0.0002,  0.0076],
          ...,
          [-0.0162,  0.0129, -0.0302,  ..., -0.0197, -0.0143,  0.0035],
          [-0.0051, -0.0330, -0.0027,  ..., -0.0348,  0.0190,  0.0001],
          [-0.0299,  0.0401,  0.0162,  ..., -0.0360, -0.0065,  0.0054]]],


        [[[ 0.0118,  0.0433,  0.0365,  ...,  0.0402,  0.0078,  0.0029],
          [ 0.0050, -0.0310,  0.0113,  ...,  0.0267, -0.0045,  0.0193],
          [-0.0196, -0.0112,  0.0022,  ..., -0.0137, -0.0400,  0.0375],
          ...,
          [ 0.0151, -0.0172,  0.0516,  ...,  0.0283, -0.0392,  0.0039],
          [-0.0136, -0.0004,  0.0151,  ..., -0.0525, -0.0084, -0.0140],
          [-0.0740,  0.0254,  0.0247,  ..., -0.0185, -0.0250,  0.0213]],

         [[ 0.0161, -0.0205, -0.0411,  ...,  0.0010,  0.0293, -0.0255],
          [-0.0252, -0.0155,  0.0183,  ...,  0.0329,  0.0010,  0.0369],
          [-0.0136, -0.0192,  0.0033,  ...,  0.0413, -0.0121,  0.0059],
          ...,
          [-0.0174,  0.0145,  0.0107,  ...,  0.0018,  0.0450, -0.0040],
          [ 0.0829,  0.0208, -0.0108,  ..., -0.0291, -0.0362, -0.0288],
          [ 0.0026, -0.0257,  0.0152,  ...,  0.0356,  0.0076, -0.0048]],

         [[-0.0098, -0.0167,  0.0031,  ..., -0.0252,  0.0185,  0.0408],
          [ 0.0396,  0.0190,  0.0127,  ...,  0.0015,  0.0048, -0.0003],
          [ 0.0453, -0.0009, -0.0130,  ...,  0.0021, -0.0186,  0.0081],
          ...,
          [ 0.0187,  0.0306,  0.0213,  ...,  0.0210,  0.0307, -0.0107],
          [-0.0172, -0.0505, -0.0307,  ...,  0.0165,  0.0270, -0.0207],
          [ 0.0369, -0.0365, -0.0053,  ...,  0.0260,  0.0132, -0.0038]]],


        [[[-0.0273,  0.0146, -0.0249,  ...,  0.0076, -0.0065,  0.0224],
          [ 0.0089, -0.0119, -0.0368,  ...,  0.0084,  0.0072,  0.0114],
          [-0.0079, -0.0172, -0.0194,  ...,  0.0057, -0.0238,  0.0199],
          ...,
          [ 0.0344,  0.0110,  0.0175,  ...,  0.0069, -0.0197,  0.0363],
          [ 0.0353, -0.0064,  0.0152,  ...,  0.0181,  0.0489, -0.0113],
          [-0.0496, -0.0119,  0.0305,  ...,  0.0204,  0.0095, -0.0072]],

         [[ 0.0194, -0.0249,  0.0267,  ..., -0.0324,  0.0048, -0.0409],
          [-0.0170,  0.0314,  0.0284,  ..., -0.0135, -0.0624,  0.0074],
          [-0.0033, -0.0315, -0.0196,  ...,  0.0169, -0.0269, -0.0469],
          ...,
          [ 0.0071,  0.0211, -0.0085,  ...,  0.0080, -0.0157, -0.0132],
          [-0.0176,  0.0330,  0.0847,  ...,  0.0636, -0.0130, -0.0436],
          [ 0.0009, -0.0204, -0.0455,  ..., -0.0104,  0.0138, -0.0272]],

         [[ 0.0014,  0.0131,  0.0109,  ..., -0.0306, -0.0080, -0.0100],
          [-0.0447,  0.0356, -0.0170,  ..., -0.0196, -0.0394, -0.0030],
          [ 0.0133,  0.0217,  0.0251,  ...,  0.0014,  0.0168, -0.0128],
          ...,
          [ 0.0195, -0.0328,  0.0026,  ...,  0.0181, -0.0079,  0.0053],
          [ 0.0066, -0.0115, -0.0058,  ..., -0.0279,  0.0086, -0.0293],
          [ 0.0155, -0.0275,  0.0301,  ..., -0.0175,  0.0125,  0.0422]]],


        ...,


        [[[ 0.0085,  0.0286, -0.0193,  ...,  0.0575,  0.0624, -0.0180],
          [-0.0121,  0.0097, -0.0041,  ..., -0.0046, -0.0297, -0.0132],
          [ 0.0193,  0.0319, -0.0011,  ...,  0.0842,  0.0024,  0.0216],
          ...,
          [ 0.0030, -0.0581,  0.0155,  ...,  0.0116, -0.0309, -0.0021],
          [ 0.0102,  0.0271, -0.0054,  ..., -0.0410,  0.0046,  0.0071],
          [-0.0105, -0.0302, -0.0786,  ..., -0.0722, -0.0223,  0.0205]],

         [[ 0.0102, -0.0060,  0.0071,  ...,  0.0557, -0.0179, -0.0071],
          [ 0.0133,  0.0215, -0.0192,  ...,  0.0336,  0.0003, -0.0370],
          [ 0.0736,  0.0302,  0.0182,  ..., -0.0492, -0.0304, -0.0157],
          ...,
          [ 0.0178,  0.0133, -0.0096,  ..., -0.0147, -0.0118,  0.0109],
          [-0.0349,  0.0127, -0.0229,  ..., -0.0048, -0.0411,  0.0271],
          [-0.0292,  0.0062,  0.0508,  ..., -0.0150, -0.0263, -0.0092]],

         [[-0.0117,  0.0185, -0.0792,  ..., -0.0329, -0.0041, -0.0155],
          [ 0.0106,  0.0026,  0.0278,  ..., -0.0145, -0.0515, -0.0069],
          [ 0.0057,  0.0027,  0.0075,  ...,  0.0102, -0.0022, -0.0311],
          ...,
          [ 0.0505,  0.0170,  0.0238,  ..., -0.0346,  0.0114, -0.0517],
          [ 0.0036,  0.0283,  0.0039,  ..., -0.0094, -0.0055,  0.0056],
          [-0.0140, -0.0227, -0.0469,  ..., -0.0151, -0.0306, -0.0058]]],


        [[[-0.0027,  0.0185, -0.0161,  ...,  0.0008,  0.0461,  0.0178],
          [ 0.0129,  0.0056, -0.0212,  ...,  0.0129,  0.0171, -0.0352],
          [-0.0362, -0.0306,  0.0026,  ...,  0.0452,  0.0111, -0.0059],
          ...,
          [-0.0104, -0.0059, -0.0304,  ..., -0.0013,  0.0077,  0.0017],
          [-0.0069, -0.0100, -0.0133,  ..., -0.0098,  0.0053,  0.0027],
          [ 0.0108,  0.0390, -0.0113,  ..., -0.0073, -0.0112,  0.0154]],

         [[-0.0212,  0.0204,  0.0258,  ..., -0.0053,  0.0176, -0.0476],
          [ 0.0125,  0.0392,  0.0196,  ..., -0.0427, -0.0190, -0.0223],
          [-0.0120,  0.0075, -0.0316,  ...,  0.0149,  0.0112,  0.0178],
          ...,
          [-0.0185, -0.0145, -0.0032,  ..., -0.0054,  0.0354,  0.0254],
          [ 0.0082, -0.0540,  0.0323,  ..., -0.0058, -0.0363,  0.0051],
          [ 0.0162,  0.0447, -0.0112,  ..., -0.0144,  0.0131, -0.0733]],

         [[-0.0316,  0.0220, -0.0081,  ..., -0.0192, -0.0206, -0.0620],
          [-0.0737, -0.0132, -0.0192,  ..., -0.0172, -0.0007, -0.0463],
          [ 0.0073, -0.0223,  0.0055,  ...,  0.0385,  0.0187,  0.0185],
          ...,
          [-0.0256, -0.0416,  0.0338,  ..., -0.0006,  0.0518, -0.0334],
          [-0.0211, -0.0207,  0.0294,  ...,  0.0350,  0.0289,  0.0049],
          [-0.0070, -0.0264, -0.0694,  ...,  0.0305, -0.0267, -0.0405]]],


        [[[ 0.0054, -0.0061, -0.0005,  ..., -0.0087,  0.0184, -0.0149],
          [-0.0015,  0.0065,  0.0272,  ...,  0.0328, -0.0278, -0.0014],
          [ 0.0153,  0.0215, -0.0143,  ...,  0.0430, -0.0258,  0.0106],
          ...,
          [ 0.0265, -0.0427, -0.0455,  ..., -0.0402, -0.0110, -0.0325],
          [-0.0201,  0.0074,  0.0058,  ..., -0.0274,  0.0552,  0.0669],
          [-0.0108,  0.0201,  0.0139,  ..., -0.0397, -0.0451, -0.0204]],

         [[-0.0199,  0.0304,  0.0293,  ..., -0.0404,  0.0075,  0.0160],
          [-0.0208,  0.0019, -0.0206,  ...,  0.0091,  0.0293, -0.0191],
          [ 0.0391, -0.0032, -0.0169,  ..., -0.0520,  0.0202, -0.0186],
          ...,
          [ 0.0179, -0.0185,  0.0187,  ...,  0.0124, -0.0003, -0.0040],
          [-0.0037, -0.0505, -0.0034,  ...,  0.0233, -0.0063,  0.0286],
          [ 0.0469,  0.0474,  0.0398,  ...,  0.0244, -0.0162, -0.0182]],

         [[ 0.0022, -0.0071,  0.0236,  ...,  0.0192, -0.0027, -0.0632],
          [ 0.0016, -0.0199,  0.0360,  ...,  0.0125, -0.0014, -0.0034],
          [-0.0044,  0.0121,  0.0031,  ...,  0.0247,  0.0131,  0.0165],
          ...,
          [-0.0087, -0.0038, -0.0167,  ..., -0.0427, -0.0289, -0.0225],
          [-0.0392,  0.0018, -0.0007,  ...,  0.0366, -0.0056,  0.0160],
          [-0.0223,  0.0245,  0.0063,  ..., -0.0044, -0.0111, -0.0155]]]],
       requires_grad=True)
```

很明显，训练过程中会伴随着大量的计算。现在，如果我们固定前6个child的参数设置为冻结的话，训练会得到很明显的加速。现在，让我们冻结到Child 6的第一个`BasicBlock`

```python
child_counter = 0
for child in model.children():
    if child_counter < 6:
        print("child ",child_counter," was frozen")
        for param in child.parameters():
            param.requires_grad = False
    elif child_counter == 6:
        children_of_child_counter = 0
        for children_of_child in child.children():
            if children_of_child_counter < 1:
                for param in children_of_child.parameters():
                    param.requires_grad = False
                print('child ', children_of_child_counter, 'of child',child_counter,' was frozen')
            else:
                print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
            children_of_child_counter += 1

    else:
        print("child ",child_counter," was not frozen")
    child_counter += 1
```

输出：
```python
child  0  was frozen
child  1  was frozen
child  2  was frozen
child  3  was frozen
child  4  was frozen
child  5  was frozen
child  0 of child 6  was frozen
child  1 of child 6  was not frozen
child  7  was not frozen
child  8  was not frozen
child  9  was not frozen
```

### 重要提示

既然已经固定了这个网络部分的参数不变，接下来要做的事情就是使它顺利跑起来。这就取决于你自己的优化器了。优化器是用来更新模型的参数的，通常，我们这么来写：

`optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)`

但是，这会给你一个错误，因为这会试图更新模型的所有参数。但是，您已经将其中一些设置为冻结!因此，只传递仍在更新的项的方法如下：

`optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)`


## 2. 模型的保存和加载

PyTorch中保存模型的主要方式有两种。建议使用“状态字典”(`state dictionaries`)。它们更快，需要更小的空间。基本上，他们不知道模型结构，他们只是参数/权重的值。所以，必须重新创建模型的结构并且载入这些参数。架构的声明与我们在上面所做的一样。

```python
# Let's assume we will save/load from a path MODEL_PATH

# Saving a Model
torch.save(model.state_dict(), MODEL_PATH)

# Loading the model.

# First create a model and define it's architecture as done above in this notebook. If you want a custom architecture.
# read below it's been covered below.
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint)
```

## 3. 修改、删除或增加最后一层

大多数使用pytorch的人（特别是如果他们用过Keras）都不喜欢这样的事实：他们不能通过`.pop()`删除最后一层。那么，让我们来看看这些事情是如何做到的。

### 修改最后一层

```python
# Load the model
model = models.resnet18(pretrained = False)

# Get number of parameters going in to the last layer. 
# we need this to change the final layer. 
num_final_in = model.fc.in_features

# The final layer of the model is model.fc so we can basically just overwrite it 
# to have the output = number of classes we need. Say, 300 classes.
NUM_CLASSES = 300
model.fc = nn.Linear(num_final_in, NUM_CLASSES)
```

### 删除最后一层 (通常，在需要一个层的参数时)

```python
# Load the model
model = models.resnet18(pretrained = False)
```

我们可以像以前一样使用`model.children()`来获取这些层。然后，我们可以使用`list()`命令将其转换为一个列表。然后，我们可以通过索引列表来删除最后一层。最后，我们可以使用PyTorch函数`nn.sequence()`将修改后的列表堆叠到一个新模型中。您可以以任何您想要的方式编辑列表。也就是说，如果你想要从第三个图层中获得图像的特征，你可以删除最后两层。您甚至可以从模型的中间删除层。但显然，这将导致不正确的数量的特征进入层后，因为大多数层改变图像的大小。在这种情况下，您可以索引模型的特定层并覆盖它，就像上面展示的那样。


```python
new_model = nn.Sequential(*list(model.children())[:-1])
```

```python
new_model_2_removed = nn.Sequential(*list(model.children())[:-2])
```

### 增加层

比方说，你想给我们现在的模型添加一个完全连接的层。一种显而易见的方法是编辑上面讨论的列表，并将其附加到另一层。然而，通常我们训练了这样一个模型，想看看是否可以加载该模型，并在其上添加一个新层。如上所述，加载的模型应该与保存的模型具有相同的体系结构，因此我们不能使用`list`方法。我们需要在上面添加图层。在PyTorch中实现这一点的方法很简单——我们只需要创建一个自定义模型!这就把我们带到了下一节——创建自定义模型。


## 4. 自定义模型 : 结合 Section 1-3，在模型头部添加层

让我们创建一个自定义模型。如上所述，我们将从一个预先训练的网络加载一半的模型。这看起来很复杂，对吧？一半的模型是经过训练的，一半是新的。此外，我们希望其中一些层固定起来。有些是可更新的。实际上，一旦您完成了这些，您就可以使用PyTorch中的模型体系结构做任何事情。



```python
# Some imports first
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torchvision import datasets, models, transforms

# New models are defined as classes. Then, when we want to create a model we create an object instantiating this class.
class Resnet_Added_Layers_Half_Frozen(nn.Module):
    def __init__(self,LOAD_VIS_URL=None):
        super(ResnetCombinedFull2, self).__init__()
    
         # Start with half the resnet model, swap out the final layer because that's the model we had defined above. 
        model = models.resnet18(pretrained = False)
        num_final_in = model.fc.in_features
        model.fc = nn.Linear(num_final_in, 300)
        
        # Now that the architecture is defined same as above, let's load the model we would have trained above. 
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint)
        
        
        # Let's freeze the same as above. Same code as above without the print statements
        child_counter = 0
        for child in model.children():
            if child_counter < 6:
                for param in child.parameters():
                    param.requires_grad = False
            elif child_counter == 6:
                children_of_child_counter = 0
                for children_of_child in child.children():
                    if children_of_child_counter < 1:
                        for param in children_of_child.parameters():
                            param.requires_grad = False
                    else:
                        children_of_child_counter += 1

            else:
                print("child ",child_counter," was not frozen")
            child_counter += 1
        
        # Now, let's define new layers that we want to add on top. 
        # Basically, these are just objects we define here. The "adding on top" is defined by the forward()
        # function which decides the flow of the input data into the model.
        
        # NOTE - Even the above model needs to be passed to self.
        self.vismodel = nn.Sequential(*list(model.children()))
        self.projective = nn.Linear(512,400)
        self.nonlinearity = nn.ReLU(inplace=True)
        self.projective2 = nn.Linear(400,300)
        
    
    # The forward function defines the flow of the input data and thus decides which layer/chunk goes on top of what.
    def forward(self,x):
        x = self.vismodel(x)
        x = torch.squeeze(x)
        x = self.projective(x)
        x = self.nonlinearity(x)
        x = self.projective2(x)
        return x
```

## 5. 自定义损失函数

现在我们的模型已经就绪，我们可以加载任何东西并创建任何我们想要的架构。这就给我们在整个流程中留下了两个重要的组件——加载数据和训练部分。让我们来看看训练部分。这一步中最重要的两个组件是优化器（optimizer）和损失函数（loss function）。损失函数量化我们现有的模型离我们想要的位置有多远，而优化器决定如何更新参数，以便我们可以最小化损失。

有时，我们需要定义自己的损失函数。这里有一些关于这个需要知道的事情：

* 自定义损失函数也是使用自定义类定义的。他们和自定义模型一样，继承了torch.nn.Module。
* 通常，我们需要改变一个输入的维度。这可以使用view()函数来完成。
* 如果我们想给张量增加一个维数，使用unsqueeze()函数。
* 最后由损失函数返回的值必须是标量值。不是一个向量/张量。

这里我展示一个定制的损失称为`Regress_Loss`，2个输入类型的输入x和y，然后它会reshape x和y，最后返回损失通过计算L2损失。这是一个经常在训练网络会遇到的事情。

假设x形状为(5,10)，y形状为(5,5,10)。因此，我们需要给x增加一个维度，然后沿着增加的维度重复来匹配y的维度，那么(x-y)就是形状(5,5,10)我们需要对所有的三个维度相加，也就是三个`torch.sum()`来得到一个标量。



```python
class Regress_Loss(torch.nn.Module):
    
    def __init__(self):
        super(Regress_Loss,self).__init__()
        
    def forward(self,x,y):
        y_shape = y.size()[1]
        x_added_dim = x.unsqueeze(1)
        x_stacked_along_dimension1 = x_added_dim.repeat(1,NUM_WORDS,1)
        diff = torch.sum((y - x_stacked_along_dimension1)**2,2)
        totloss = torch.sum(torch.sum(torch.sum(diff)))
        return totloss
```
