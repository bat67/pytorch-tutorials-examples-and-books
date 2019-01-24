- [PyTorch for Numpy users.](#pytorch-for-numpy-users)
  - [Types](#types)
  - [Constructors](#constructors)
    - [Ones and zeros](#ones-and-zeros)
    - [From existing data](#from-existing-data)
    - [Numerical ranges](#numerical-ranges)
    - [Linear algebra](#linear-algebra)
    - [Building matrices](#building-matrices)
    - [Attributes](#attributes)
    - [Indexing](#indexing)
    - [Shape manipulation](#shape-manipulation)
    - [Item selection and manipulation](#item-selection-and-manipulation)
    - [Calculation](#calculation)
    - [Arithmetic and comparison operations](#arithmetic-and-comparison-operations)
    - [Random numbers](#random-numbers)
    - [Numerical operations](#numerical-operations)


# PyTorch for Numpy users.


## Types

| Numpy      | PyTorch                       |
| ---------- | ----------------------------- |
| np.ndarray | torch.Tensor                  |
| np.float32 | torch.float32<br>torch.float  |
| np.float64 | torch.float64<br>torch.double |
| np.float16 | torch.float16<br>torch.half   |
| np.int8    | torch.int8                    |
| np.uint8   | torch.uint8                   |
| np.int16   | torch.int16<br>torch.short    |
| np.int32   | torch.int32<br>torch.int      |
| np.int64   | torch.int64<br>torch.long     |



## Constructors

### Ones and zeros

| Numpy            | PyTorch             |
| ---------------- | ------------------- |
| np.empty((2, 3)) | torch.empty(2, 3)   |
| np.empty_like(x) | torch.empty_like(x) |
| np.eye           | torch.eye           |
| np.identity      | torch.eye           |
| np.ones          | torch.ones          |
| np.ones_like     | torch.ones_like     |
| np.zeros         | torch.zeros         |
| np.zeros_like    | torch.zeros_like    |


### From existing data

| Numpy                                                         | PyTorch                                       |
| ------------------------------------------------------------- | --------------------------------------------- |
| np.array([[1, 2], [3, 4]])                                    | torch.tensor([[1, 2], [3, 4]])                |
| np.array([3.2, 4.3], dtype=np.float16)<br>np.float16([3.2, 4.3]) | torch.tensor([3.2, 4.3], dtype=torch.float16) |
| x.copy()                                                      | x.clone()                                     |
| np.fromfile(file)                                             | torch.tensor(torch.Storage(file))             |
| np.frombuffer                                                 |                                               |
| np.fromfunction                                               |                                               |
| np.fromiter                                                   |                                               |
| np.fromstring                                                 |                                               |
| np.load                                                       | torch.load                                    |
| np.loadtxt                                                    |                                               |
| np.concatenate                                                | torch.cat                                     |


### Numerical ranges

| Numpy                | PyTorch                 |
| -------------------- | ----------------------- |
| np.arange(10)        | torch.arange(10)        |
| np.arange(2, 3, 0.1) | torch.arange(2, 3, 0.1) |
| np.linspace          | torch.linspace          |
| np.logspace          | torch.logspace          |


### Linear algebra

| Numpy  | PyTorch  |
| ------ | -------- |
| np.dot | torch.mm |


### Building matrices

| Numpy   | PyTorch    |
| ------- | ---------- |
| np.diag | torch.diag |
| np.tril | torch.tril |
| np.triu | torch.triu |


### Attributes

| Numpy     | PyTorch      |
| --------- | ------------ |
| x.shape   | x.shape      |
| x.strides | x.stride()   |
| x.ndim    | x.dim()      |
| x.data    | x.data       |
| x.size    | x.nelement() |
| x.dtype   | x.dtype      |


### Indexing

| Numpy               | PyTorch                                  |
| ------------------- | ---------------------------------------- |
| x[0]                | x[0]                                     |
| x[:, 0]             | x[:, 0]                                  |
| x[indices]          | x[indices]                               |
| np.take(x, indices) | torch.take(x, torch.LongTensor(indices)) |
| x[x != 0]           | x[x != 0]                                |


### Shape manipulation

| Numpy                                  | PyTorch                  |
| -------------------------------------- | ------------------------ |
| x.reshape                              | x.reshape; x.view        |
| x.resize()                             | x.resize_                |
|                                        | x.resize_as_             |
| x.transpose                            | x.transpose<br> x.permute |
| x.flatten                              | x.view(-1)               |
| x.squeeze()                            | x.squeeze()              |
| x[:, np.newaxis]<br>np.expand_dims(x, 1) | x.unsqueeze(1)           |


### Item selection and manipulation

| Numpy                                                          | PyTorch                                                                                                                                              |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| np.put                                                         | -                                                                                                                                                    |
| x.put                                                          | x.put_                                                                                                                                               |
| x = np.array([1, 2, 3])<br>x.repeat(2)<br># [1, 1, 2, 2, 3, 3] | x = torch.tensor([1, 2, 3])<br>x.repeat(2)<br># [1, 2, 3, 1, 2, 3]<br>x.repeat(2).reshape(2, -1).transpose(1, 0).reshape(-1)<br># [1, 1, 2, 2, 3, 3] |
| np.tile(x, (3, 2))                                             | x.repeat(3, 2)                                                                                                                                       |
| np.choose                                                      |                                                                                                                                                      |
| np.sort                                                        | sorted, indices = torch.sort(x, [dim])                                                                                                               |
| np.argsort                                                     | sorted, indices = torch.sort(x, [dim])                                                                                                               |
| np.nonzero                                                     | torch.nonzero                                                                                                                                        |
| np.where                                                       | torch.where                                                                                                                                          |
| x[::-1]                                                        | torch.flip(x, [0])                                                                                                                                   |


### Calculation

| Numpy         | PyTorch                        |
| ------------- | ------------------------------ |
| x.min         | x.min                          |
| x.argmin      | x.argmin                       |
| x.max         | x.max                          |
| x.argmax      | x.argmax                       |
| x.clip        | x.clamp                        |
| x.round       | x.round                        |
| np.floor(x)   | torch.floor(x); x.floor()      |
| np.ceil(x)    | torch.ceil(x); x.ceil()        |
| x.trace       | x.trace                        |
| x.sum         | x.sum                          |
| x.sum(axis=0) | x.sum(0)                       |
| x.cumsum      | x.cumsum                       |
| x.mean        | x.mean                         |
| x.std         | x.std                          |
| x.prod        | x.prod                         |
| x.cumprod     | x.cumprod                      |
| x.all         | (x == 1).sum() == x.nelement() |
| x.any         | (x == 1).sum() > 0             |


### Arithmetic and comparison operations

| Numpy            | PyTorch |
| ---------------- | ------- |
| np.less          | x.lt    |
| np.less_equal    | x.le    |
| np.greater       | x.gt    |
| np.greater_equal | x.ge    |
| np.equal         | x.eq    |
| np.not_equal     | x.ne    |

### Random numbers

| Numpy                    | PyTorch           |
| ------------------------ | ----------------- |
| np.random.seed           | torch.manual_seed |
| np.random.permutation(5) | torch.randperm(5) |


### Numerical operations

| Numpy   | PyTorch    |
| ------- | ---------- |
| np.sign | torch.sign |
| np.sqrt | torch.sqrt |


