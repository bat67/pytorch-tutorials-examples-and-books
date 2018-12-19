from visdom import Visdom
import numpy as np

vis = Visdom()

vis.histogram(X=np.random.rand(10000), opts=dict(numbins=20))
