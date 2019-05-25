from visdom import Visdom
import numpy as np

vis = Visdom()

# 2D scatterplot with custom intensities (red channel)
vis.scatter(
    X =  np.random.rand(255, 2),
    Y = (np.random.randn(255) > 0) + 1 ,
   opts=dict(
        markersize=10,
        markercolor=np.floor(np.random.random((2, 3)) * 255),
	legend=['Men', 'Women']
    ),
)
