- [PyTorch 概览](#pytorch-概览)
  - [导入相关包](#导入相关包)
  - [配置](#配置)
  - [数据](#数据)
  - [模型](#模型)
  - [训练](#训练)
  - [测试](#测试)
  - [其他](#其他)


# PyTorch 概览


[PyTorch](https://pytorch.org/) is a flexible deep learning framework that allows automatic differentiation through dynamic neural networks (i.e., networks that utilise dynamic control flow like if statements and while loops). It supports GPU acceleration, [distributed training](https://pytorch.org/docs/stable/distributed.html), [various optimisations](https://pytorch.org/2018/05/02/road-to-1.0.html), and plenty more neat features. These are some notes on how I think about using PyTorch, and don't encompass all parts of the library or every best practice, but may be helpful to others.

Neural networks are a subclass of *computation graphs*. Computation graphs receive input data, and data is routed to and possibly transformed by nodes which perform processing on the data. In deep learning, the neurons (nodes) in neural networks typically transform data with parameters and differentiable functions, such that the parameters can be optimised to minimise a loss via gradient descent. More broadly, the functions can be stochastic, and the structure of the graph can be dynamic. So while neural networks may be a good fit for [dataflow programming](https://en.wikipedia.org/wiki/Dataflow_programming), PyTorch's API has instead centred around [imperative programming](https://en.wikipedia.org/wiki/Imperative_programming), which is a more common way for thinking about programs. This makes it easier to read code and reason about complex programs, without necessarily sacrificing much performance; PyTorch is actually pretty fast, with plenty of optimisations that you can safely forget about as an end user (but you can dig in if you really want to).

The rest of this document, based on the [official MNIST example](https://github.com/pytorch/examples/tree/master/mnist), is about *grokking* PyTorch, and should only be looked at after the [official beginner tutorials](https://pytorch.org/tutorials/). For readability, the code is presented in chunks interspersed with comments, and hence not separated into different functions/files as it would normally be for clean, modular code.

## 导入相关包

```py
import argparse
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

These are pretty standard imports, with the exception of the `torchvision` modules that are used for computer vision problems in particular.

## 配置

```py
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before checkpointing')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(args.seed)
if use_cuda:
  torch.cuda.manual_seed(args.seed)
```

`argparse` is a standard way of dealing with command-line arguments in Python.

A good way to write device-agnostic code (benefitting from GPU acceleration when available but falling back to CPU when not) is to pick and save the appropriate `torch.device`, which can be used to determine where tensors should be stored. See the [official docs](https://pytorch.org/docs/master/notes/cuda.html#device-agnostic-code) for more tips on device-agnostic code. The PyTorch way is to put device placement under the control of the user, which may seem a nuisance for simple examples, but makes it much easier to work out where tensors are - which is useful for a) debugging and b) making efficient use of devices manually.

For repeatable experiments, it is necessary to set random seeds for anything that uses random number generation (including `random` or `numpy` if those are used too). Note that cuDNN uses nondeterministic algorithms, and it can be disabled using `torch.backends.cudnn.enabled = False`.

## 数据

```py
data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')
train_data = datasets.MNIST(data_path, train=True, download=True,
                            transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))]))
test_data = datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = DataLoader(train_data, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size,
                         num_workers=4, pin_memory=True)
```

Since `torchvision` models get stored under `~/.torch/models/`, I like to store `torchvision` datasets under `~/.torch/datasets`. This is my own convention, but makes it easier if you have lots of projects that depend on MNIST, CIFAR-10 etc. In general it's worth keeping datasets separately to code if you end up reusing several datasets.

`torchvision.transforms` contains lots of handy transformations for single images, such as cropping and normalisation.

`DataLoader` contains many options, but beyond `batch_size` and `shuffle`, `num_workers` and `pin_memory` are worth knowing for efficiency. `num_workers` > 0 uses subprocesses to asynchronously load data, rather than making the main process block on this. The typical use-case is when loading data (e.g. images) from disk and maybe transforming them too - this can be done in parallel with the network processing the data. You will want to tune the amount to a) minimise the number of workers and hence CPU and RAM usage (each worker loads a separate batch, not individual samples within a batch) b) minimise the time the network is waiting for data. `pin_memory` uses [pinned memory](https://pytorch.org/docs/master/notes/cuda.html#use-pinned-memory-buffers) (as opposed to paged memory) to speed up any RAM to GPU transfers (and does nothing for CPU-only code).

## 模型

```py
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

model = Net().to(device)
optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

if args.resume:
  model.load_state_dict(torch.load('model.pth'))
  optimiser.load_state_dict(torch.load('optimiser.pth'))
```

Network initialisation typically includes member variables, layers which contain trainable parameters, and maybe separate trainable parameters and non-trainable buffers. The forward pass then uses these in conjunction with functions from `F` that are purely functional (don't contain parameters). Some people prefer to have completely functional networks (e.g., keeping parameters separately and using `F.conv2d` instead of `nn.Conv2d`) or networks completely made of layers (e.g., `nn.ReLU` instead of `F.relu`).

`.to(device)` is a convenient way of sending the device parameters (and buffers) to GPU if `device` is set to GPU, doing nothing otherwise (when `device` is set to CPU). It's important to transfer the network parameters to the appropriate device before passing them to the optimiser, otherwise the optimiser will not be keeping track of the parameters properly!

Both neural networks (`nn.Module`) and optimisers (`optim.Optimizer`) have the ability to save and load their internal state, and `.load_state_dict(state_dict)` is the recommended way to do so - you'll want to reload the state of both to resume training from previously saved state dictionaries. Saving the entire object can be [error prone](https://pytorch.org/docs/stable/notes/serialization.html). If you have saved tensors on GPU and want to load them on CPU or another GPU, the easiest way is to directly load them onto CPU using the [`map_location` option](https://pytorch.org/docs/stable/torch.html#torch.load), e.g., `torch.load('model.pth', map_location='cpu')`.

Some points of note not shown here are that the forward pass can make use of control flow (e.g., a member variable or even the data itself can determine the execution of an if statement. It is also perfectly valid to `print` tensors in the middle, making debugging much easier. Finally, the forward pass can make use of multiple arguments. A short snippet (not tied to any sensible idea) to illustrate this is below:

```py
def forward(self, x, hx, drop=False):
  hx2 = self.rnn(x, hx)
  print(hx.mean().item(), hx.var().item())
  if hx.max.item() > 10 or self.can_drop and drop:
    return hx
  else:
    return hx2
```

## 训练

```py
model.train()
train_losses = []

for i, (data, target) in enumerate(train_loader):
  data = data.to(device=device, non_blocking=True)
  target = target.to(device=device, non_blocking=True)
  optimiser.zero_grad()
  output = model(data)
  loss = F.nll_loss(output, target)
  loss.backward()
  train_losses.append(loss.item())
  optimiser.step()

  if i % 10 == 0:
    print(i, loss.item())
    torch.save(model.state_dict(), 'model.pth')
    torch.save(optimiser.state_dict(), 'optimiser.pth')
    torch.save(train_losses, 'train_losses.pth')
```

Network modules are by default set to training mode - which impacts the way some modules work, most noticeably dropout and batch normalisation. It's best to set this manually anyway with `.train()`, which propagates the training flag down all children modules.

Here the `.to()` method not only takes the device, but also sets `non_blocking=True`, which enables asynchronous data copies to GPU from pinned memory, hence allowing the CPU to keep operating during the transfer; `non_blocking=True` is simply a no-op otherwise.

Before collecting a new set of gradients with `loss.backward()` and doing backpropagation with `optimiser.step()`, it's necessary to manually zero the gradients of the parameters being optimised with `optimiser.zero_grad()`. By default, PyTorch *accumulates* gradients, which is very handy when you don't have enough resources to calculate all the gradients you need in one go.

PyTorch uses a tape-based automatic gradient (autograd) system - it collects which operations were done on tensors in order, and then replays them backwards to do reverse-mode differentiation. This is why it is super flexible and allows arbitrary computation graphs. If none of the tensors require gradients (you'd have to set `requires_grad=True` when constructing a tensor for this) then no graph is stored! However, networks tend to have parameters that require gradients, so any computation done from the output of a network will be stored in the graph. So if you want to store data resulting from this, you'll need to manually disable gradients or, more commonly, store it as a Python number (via `.item()` on a PyTorch scalar) or numpy array. See the [official docs](https://pytorch.org/docs/stable/notes/autograd.html) for more on autograd.

One way to cut the computation graph is to use `.detach()`, which you may use when passing on a hidden state when training RNNs with truncated backpropagation-through-time. It's also handy when differentiating a loss where one component is the output of another network, but this other network shouldn't be optimised with respect to the loss - examples include training a discriminator from a generator's outputs in GAN training, or training the policy of an actor-critic algorithm using the value function as a baseline (e.g. A2C). Another technique for preventing gradient calculations that is efficient in GAN training (training the generator from the discriminator) and typical in fine-tuning is to loop through a networks parameters and set `param.requires_grad = False`.

Apart from logging results in the console/in a log file, it's important to checkpoint model parameters (and optimiser state) just in case. You can also use `torch.save()` to save normal Python objects, but other standard choices include the built-in `pickle`.

## 测试

```py
model.eval()
test_loss, correct = 0, 0

with torch.no_grad():
  for data, target in test_loader:
    data = data.to(device=device, non_blocking=True)
    target = target.to(device=device, non_blocking=True)
    output = model(data)
    test_loss += F.nll_loss(output, target, reduction='sum').item()
    pred = output.argmax(1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_data)
acc = correct / len(test_data)
print(acc, test_loss)
```

In response to `.train()` earlier, networks should explicitly be set to evaluation mode using `.eval()`.

As mentioned previously, the computation graph would normally be made when using a network. By using the `no_grad` context manager via `with torch.no_grad()` this is prevented from happening.

## 其他

This is an extra section just to add a few useful asides.

Memory problems? Check the [official docs](https://pytorch.org/docs/stable/notes/faq.html#my-model-reports-cuda-runtime-error-2-out-of-memory) for tips.

CUDA errors? They are a pain to debug, and are usually a logic problem that would come up with a more intelligible error message on CPU. It's best to be able to easily switch between CPU and GPU if you are planning on using the GPU. A more general development tip is to be able to set up your code so that it's possible to run through all of the logic quickly to check it before launching a proper job - examples would be preparing a small/synthetic dataset, running one train + test epoch, etc. If it is a CUDA error, or you really can't switch to CPU, setting ` CUDA_LAUNCH_BLOCKING=1` will make CUDA kernel launches synchronous and as a result provide better error messages.

A note for `torch.multiprocessing`, or even just running multiple PyTorch scripts at once. Because PyTorch uses multithreaded BLAS libraries to speed up linear algebra computations on CPU, it'll typically use several cores. If you want to run several things at once, with multiprocessing or several scripts, it may be useful to manually reduce these by setting the environment variable `OMP_NUM_THREADS` to 1 or another small number - this reduces the chance of CPU thrashing. The [official docs](https://pytorch.org/docs/stable/notes/multiprocessing.html) have some other notes for multiprocessing in particular.
