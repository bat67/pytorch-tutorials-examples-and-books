- [PyTorch版本变化及迁移指南](#pytorch版本变化及迁移指南)
  - [主要区别：](#主要区别)
  - [1、合并Tensor和Variable和类](#1合并tensor和variable和类)
    - [1.1、`Tensor`中的`type()`改变了](#11tensor中的type改变了)
    - [1.2、什么时候`autograd`开始自动求导？](#12什么时候autograd开始自动求导)
    - [1.3、操作requires_grad标志](#13操作requires_grad标志)
    - [1.4、关于`.data`](#14关于data)
  - [2、现在一些操作返回0维（标量）Tensors](#2现在一些操作返回0维标量tensors)
    - [2.1、积累损失](#21积累损失)
  - [3、弃用`volatile`标签](#3弃用volatile标签)
  - [4、`dtypes`，`devices`和NumPy风格的创作功能](#4dtypesdevices和numpy风格的创作功能)
    - [4.1 Tensor Attributes](#41-tensor-attributes)
      - [4.1.1 `torch.dtype`](#411-torchdtype)
      - [4.1.2 `torch.device`](#412-torchdevice)
      - [4.1.3 `torch.layout`](#413-torchlayout)
    - [4.2 创建 Tensors](#42-创建-tensors)
  - [5、编写device-agnostic代码](#5编写device-agnostic代码)
  - [6、nn.Module中子模块名称，参数和缓冲区中的新边界约束](#6nnmodule中子模块名称参数和缓冲区中的新边界约束)
    - [6.1、代码示例（将它们放在一起）](#61代码示例将它们放在一起)



# PyTorch版本变化及迁移指南

> 确实有一些变化的，而且现在好多pytorch的书籍还都是旧的版本，好多代码已经无法运行，是时候看看版本变化记录了！


## 主要区别：

* Tensors并Variables已合并
* Tensors支持0维（标量）
* 弃用volatile标签
* dtypes，devices和NumPy风格的创作功能
* 编写device-agnostic代码
* nn.Module中子模块名称，参数和缓冲区中的新边界约束

## 1、合并Tensor和Variable和类

`torch.autograd.Variable`和`torch.Tensor`现在类相同。确切地说，`torch.Tensor`能够像`Variable`一样自动求导；`Variable`继续像以前一样工作但返回一个`torch.Tensor`类型的对象。意味着你在代码中不再需要`Variable`包装器。

### 1.1、`Tensor`中的`type()`改变了

`type()`不再反映张量的数据类型。使用`isinstance()`或`x.type()`替代：

```python
    >>> x = torch.DoubleTensor([1, 1, 1])
    >>> print(type(x))  # was torch.DoubleTensor
    "<class 'torch.Tensor'>"
    >>> print(x.type())  # OK: 'torch.DoubleTensor'
    'torch.DoubleTensor'
    >>> print(isinstance(x, torch.DoubleTensor))  # OK: True
    True
```

### 1.2、什么时候`autograd`开始自动求导？

`requires_grad`是`autograd`的核心标志，现在是`Tensors`上的一个属性。让我们看看在代码中如何体现的。

`autograd`使用以前用于`Variables`的相同规则。当张量定义了`requires_grad=True`就可以自动求导了。例如，

```python
    >>> x = torch.ones(1)  # create a tensor with requires_grad=False (default)
    >>> x.requires_grad
    False
    >>> y = torch.ones(1)  # another tensor with requires_grad=False
    >>> z = x + y
    >>> # both inputs have requires_grad=False. so does the output
    >>> z.requires_grad
    False
    >>> # then autograd won't track this computation. let's verify!
    >>> z.backward()
    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    >>>
    >>> # now create a tensor with requires_grad=True
    >>> w = torch.ones(1, requires_grad=True)
    >>> w.requires_grad
    True
    >>> # add to the previous result that has require_grad=False
    >>> total = w + z
    >>> # the total sum now requires grad!
    >>> total.requires_grad
    True
    >>> # autograd can compute the gradients as well
    >>> total.backward()
    >>> w.grad
    tensor([ 1.])
    >>> # and no computation is wasted to compute gradients for x, y and z, which don't require grad
    >>> z.grad == x.grad == y.grad == None
    True
```

### 1.3、操作requires_grad标志

除了直接设置属性之外，您可以使用`my_tensor.requiresgrad(requires_grad=True)`直接修改此标志，或者如上例所示，在创建时将其作为参数传递（默认为`False`），例如:

```python
    >>> existing_tensor.requires_grad_()
    >>> existing_tensor.requires_grad
    True
    >>> my_tensor = torch.zeros(3, 4, requires_grad=True)
    >>> my_tensor.requires_grad
    True
```

### 1.4、关于`.data`

`.data`是从`Variable`中获取`Tensor`的方法。合并后，调用`y = x.data`仍然具有类似的语义。因此y将是与x共享同的`Tensor`相数据，x与计算历史无关，并具有`requires_grad=False`。

但是，`.data`在某些情况下可能不安全。`x.data`上的任何变化都不会被`autograd`跟踪，并且x在向后传递中计算梯度将不正确。一种更安全的替代方法是使用`x.detach()`，它也返回一个`Tensor`与`requires_grad=False`共享数据的数据，但是如果x需要反向传播那就会使用`autograd`直接改变报告。

下面是一个`.data`和`x.detach()`（以及为什么我们建议detach一般使用）之间的区别的例子。

如果你使用`Tensor.detach()`，保证梯度计算是正确的。

```python
    >>> a = torch.tensor([1,2,3.], requires_grad = True)
    >>> out = a.sigmoid()
    >>> c = out.detach()
    >>> c.zero_()
    tensor([ 0.,  0.,  0.])
     
    >>> out  # modified by c.zero_() !!
    tensor([ 0.,  0.,  0.])
     
    >>> out.sum().backward()  # Requires the original value of out, but that was overwritten by c.zero_()
    RuntimeError: one of the variables needed for gradient computation has been modified by an 
```

然而，使用Tensor.data可能是不安全的，并且当梯度计算需要张量但直接修改时可能容易导致不正确的梯度。

```python
    >>> a = torch.tensor([1,2,3.], requires_grad = True)
    >>> out = a.sigmoid()
    >>> c = out.data
    >>> c.zero_()
    tensor([ 0.,  0.,  0.])
     
    >>> out  # out  was modified by c.zero_()
    tensor([ 0.,  0.,  0.])
     
    >>> out.sum().backward()
    >>> a.grad  # The result is very, very wrong because `out` changed!
    tensor([ 0.,  0.,  0.])
```

## 2、现在一些操作返回0维（标量）Tensors

以前，Tensor向量（1维张量）的索引返回一个Python数字，但是Variable的索引向量返回一个`(1,)`的向量！即tensor.sum()返回一个Python数字，但`variable.sum()`会返回一个大小为`(1,)`的向量。

幸运的是，此版本在PyTorch中引入了适当的标量（0维张量）支持！可以使用新torch.tensor函数来创建标量（稍后会对其进行更详细的解释;现在只需将它看作PyTorch中与numpy.array的等价物）。现在你可以做这样的事情：

```python
    >>> torch.tensor(3.1416)         # create a scalar directly
    tensor(3.1416)
    >>> torch.tensor(3.1416).size()  # scalar is 0-dimensional
    torch.Size([])
    >>> torch.tensor([3]).size()     # compare to a vector of size 1
    torch.Size([1])
    >>>
    >>> vector = torch.arange(2, 6)  # this is a vector
    >>> vector
    tensor([ 2.,  3.,  4.,  5.])
    >>> vector.size()
    torch.Size([4])
    >>> vector[3]                    # indexing into a vector gives a scalar
    tensor(5.)
    >>> vector[3].item()             # .item() gives the value as a Python number
    5.0
    >>> mysum = torch.tensor([2, 3]).sum()
    >>> mysum
    tensor(5)
    >>> mysum.size()
    torch.Size([])
```


### 2.1、积累损失

考虑到经常使用的`total_loss += loss.data[0]`。0.4.0之前。loss是(1,)张量的Variable包装器，但在0.4.0中loss现在是一个0尺寸标量。标量索引是没有意义的（目前只提出一个警告，但在0.5.0中将会报错）。loss.item()用于从标量中获取Python数字。

请注意，如果您在累积损失时未将其转换为Python数字，则可能会发现程序中的内存使用量增加。这是因为上面表达式的右侧曾经是一个Python浮点数，而现在它是一个0的张量。因此，总损失累积了张量和它们的历史梯度，可能导致巨大的autograd图形不必要的保存大量时间。


## 3、弃用`volatile`标签

`volatile`标签现在已被弃用，不起作用。以前，任何涉及Variable的计算`volatile=True`都不会被跟踪autograd。这已经被换成了一套更加灵活的上下文管理的，包括`torch.no_grad()`，`torch.set_grad_enabled(grad_mode)`及其他。

```python
    >>> x = torch.zeros(1, requires_grad=True)
    >>> with torch.no_grad():
    ...     y = x * 2
    >>> y.requires_grad
    False
    >>>
    >>> is_train = False
    >>> with torch.set_grad_enabled(is_train):
    ...     y = x * 2
    >>> y.requires_grad
    False
    >>> torch.set_grad_enabled(True)  # this can also be used as a function
    >>> y = x * 2
    >>> y.requires_grad
    True
    >>> torch.set_grad_enabled(False)
    >>> y = x * 2
    >>> y.requires_grad
    False
```

## 4、`dtypes`，`devices`和NumPy风格的创作功能

在以前的PyTorch版本中，我们用来指定的数据类型（例如float vs double），设备类型（cpu vs cuda）和layout（dense vs sparse）作为"张量类型"。例如，torch.cuda.sparse.DoubleTensor是Tensor的double数据类型，在CUDA设备只能够，以及配备COO稀疏张量layout。

在此版本中，我们引入torch.dtype，torch.device以及torch.layout类，允许通过NumPy的风格创建这些属性的功能进行更好的管理。

pytorch从0.4开始提出了Tensor Attributes，主要包含了`torch.dtype`, `torch.device`, `torch.layout`。pytorch可以使用他们管理数据类型属性。具体可以查看Tensor Attributes

### 4.1 Tensor Attributes

* `torch.dtype`
* `torch.device`
* `torch.layout`

每个`torch.Tensor`都有`torch.dtype`, `torch.device`,和`torch.layout`。

#### 4.1.1 `torch.dtype`

torch.dtype是表示torch.Tensor的数据类型的对象。PyTorch有八种不同的数据类型：

|Data type 	|dtype 	|Tensor types|
|---------|---------|---------|
|32-bit floating point 	|torch.float32 or torch.float | torch.*.FloatTensor
|64-bit floating point 	|torch.float64 or torch.double| torch.*.DoubleTensor
|16-bit floating point 	|torch.float16 or torch.half | torch.*.HalfTensor
|8-bit integer (unsigned) 	|torch.uint8 	|torch.*.ByteTensor
|8-bit integer (signed) 	|torch.int8 	|torch.*.CharTensor
|16-bit integer (signed) 	|torch.int16 or torch.short |	torch.*.ShortTensor
|32-bit integer (signed) 	|torch.int32 or torch.int 	|torch.*.IntTensor
|64-bit integer (signed) 	|torch.int64 or torch.long 	|torch.*.LongTensor

使用方法：

```python
    >>> x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> print x.type()
    torch.FloatTensor
```

#### 4.1.2 `torch.device`

`torch.device`代表将`torch.Tensor`分配到的设备的对象。

`torch.device`包含一个设备类型（'cpu'或'cuda'设备类型）和可选的设备的序号。如果设备序号不存在，则为当前设备; 例如，torch.Tensor用设备构建'cuda'的结果等同于'cuda:X',其中X是torch.cuda.current_device()的结果。

torch.Tensor的设备可以通过Tensor.device访问属性。

构造torch.device可以通过字符串/字符串和设备编号。

通过一个字符串：

```python
    >>> torch.device('cuda:0')
    device(type='cuda', index=0)
     
    >>> torch.device('cpu')
    device(type='cpu')
     
    >>> torch.device('cuda')  # current cuda device
    device(type='cuda')
```

通过字符串和设备序号：

```python
    >>> torch.device('cuda', 0)
    device(type='cuda', index=0)
     
    >>> torch.device('cpu', 0)
    device(type='cpu', index=0)

    注意 torch.device函数中的参数通常可以用一个字符串替代。这允许使用代码快速构建原型。

        >> # Example of a function that takes in a torch.device
        >> cuda1 = torch.device('cuda:1')
        >> torch.randn((2,3), device=cuda1)

        >> # You can substitute the torch.device with a string
        >> torch.randn((2,3), 'cuda:1')
```
    注意 出于传统原因，可以通过单个设备序号构建设备，将其视为cuda设备。这匹配Tensor.get_device()，它为cuda张量返回一个序数，并且不支持cpu张量。

```python
        >> torch.device(1)
        device(type='cuda', index=1)

    注意 指定设备的方法可以使用（properly formatted）字符串或（legacy）整数型设备序数，即以下示例均等效：

        >> torch.randn((2,3), device=torch.device('cuda:1'))
        >> torch.randn((2,3), device='cuda:1')
        >> torch.randn((2,3), device=1)  # legacy
```

#### 4.1.3 `torch.layout`

`torch.layout`表示`torch.Tensor`内存布局的对象。目前，我们支持torch.strided(dense Tensors)并为torch.sparse_coo(sparse COO Tensors)提供实验支持。

torch.strided代表密集张量，是最常用的内存布局。每个strided张量都会关联 一个torch.Storage，它保存着它的数据。这些张力提供了多维度， 存储的strided视图。Strides是一个整数型列表：k-th stride表示在张量的第k维从一个元素跳转到下一个元素所需的内存。这个概念使得可以有效地执行多张量。

例：

```python
    >>> x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)
     
    >>> x.t().stride()
    (1, 5)

```

### 4.2 创建 Tensors

创造一个方法Tensor，现在也可使用dtype，device，layout，和requires_grad选项来指定返回所需的Tensor属性。例如:

```python
    >>> device = torch.device("cuda:1")
    >>> x = torch.randn(3, 3, dtype=torch.float64, device=device)
    tensor([[-0.6344,  0.8562, -1.2758],
            [ 0.8414,  1.7962,  1.0589],
            [-0.1369, -1.0462, -0.4373]], dtype=torch.float64, device='cuda:1')
    >>> x.requires_grad  # default is False
    False
    >>> x = torch.zeros(3, requires_grad=True)
    >>> x.requires_grad
    True
```

## 5、编写device-agnostic代码

以前版本的PyTorch编写device-agnostic代码非常困难（即，在不修改代码的情况下在CUDA可以使用或者只能使用CPU的设备上运行）。

device-agnostic，即设备无关，可以理解为无论什么设备都可以运行您编写的代码。

PyTorch 0.4.0及以上通过两种方法使代码兼容变得非常容易：

* 张量的device属性为所有张量提供了torch.device设备。（注意:get_device仅适用于CUDA张量）
* to方法Tensors和Modules可用于容易地将对象移动到不同的设备（代替以前的cpu()或cuda()方法）

我们推荐以下模式：

```python
    # 开始脚本，创建一个张量
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
    ...
     
    # 但是无论你获得一个新的Tensor或者Module
    # 如果他们已经在目标设备上则不会执行复制操作
    input = data.to(device)
    model = MyModule(...).to(device)
```
 
## 6、nn.Module中子模块名称，参数和缓冲区中的新边界约束

name这是一个空字符串或包含"."不再被允许进入module.add_module(name, value)，module.add_parameter(name, value)或者module.add_buffer(name, value)因为这些名称可能会在state_dict中导致数据丢失。如果您为包含这些名称的模块加载checkpoint，请在加载之前更新模块定义并进行修补state_dict。

### 6.1、代码示例（将它们放在一起）

为了方便对比0.4.0中整体推荐的变化的特征，我们来看一个0.3.1和0.4.0中常见代码模式的简单例子：

0.3.1（旧）：

```python
    model = MyRNN()
    if use_cuda:
        model = model.cuda()
     
    # train
    total_loss = 0
    for input, target in train_loader:
        input, target = Variable(input), Variable(target)
        hidden = Variable(torch.zeros(*h_shape))  # init hidden
        if use_cuda:
            input, target, hidden = input.cuda(), target.cuda(), hidden.cuda()
        ...  # get loss and optimize
        total_loss += loss.data[0]
     
    # evaluate
    for input, target in test_loader:
        input = Variable(input, volatile=True)
        if use_cuda:
            ...
        ...
```

0.4.0（新）：

```python
    # torch.device object used throughout this script
    device = torch.device("cuda" if use_cuda else "cpu")
     
    model = MyRNN().to(device)
     
    # train
    total_loss = 0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        hidden = input.new_zeros(*h_shape)  # has the same device & dtype as `input`
        ...  # get loss and optimize
        total_loss += loss.item()           # get Python number from 1-element Tensor
     
    # evaluate
    with torch.no_grad():                   # operations inside don't track history
        for input, target in test_loader:
            ...

```

感谢您的阅读！有关更多详细信息，请参阅官方文档和发行说明。

Happy PyTorching！
