from visdom import Visdom
import numpy as np

vis = Visdom()

# boxplot
X = np.random.rand(100, 2)
X[:, 1] += 2

vis.boxplot(
    X=X,
    opts=dict(legend=['Men', 'Women'])
)
