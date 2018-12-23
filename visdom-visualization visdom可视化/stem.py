from visdom import Visdom
import numpy as np
import math

vis = Visdom()

# stemplot
Y = np.linspace(0, 2 * math.pi, 70)
X = np.column_stack((np.sin(Y), np.cos(Y)))
vis.stem(
    X=X,
    Y=Y,
    opts=dict(legend=['Sine', 'Cosine'])
)
