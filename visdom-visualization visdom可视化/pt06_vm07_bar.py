from visdom import Visdom
import numpy as np
import math

vis = Visdom()

# 单个条形图
vis.bar(X=np.random.rand(20))

# 堆叠条形图
vis.bar(
    X=np.abs(np.random.rand(5, 3)),
    opts=dict(
        stacked=True,
        legend=['Sina', '163', 'AliBaBa'],
        rownames=['2013', '2014', '2015', '2016', '2017']
    )
)

# 分组条形图
vis.bar(
    X=np.random.rand(20, 3),
    opts=dict(
        stacked=False,
        legend=['A', 'B', 'C']
    )
)
