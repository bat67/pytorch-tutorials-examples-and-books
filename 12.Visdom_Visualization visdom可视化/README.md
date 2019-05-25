# visdom基本概念

* Panes（窗格）

* Environment（环境）

* State（状态）

# 使用

安装visdom

`pip install visdom`

启动服务器

`python -m visdom.server`

访问visdom

`http://loacalhost:8097`

终端输入

`python text.py`

（或者打开 `ipython`，输入 `run text.py`）

# 可视化接口

例子见本文件夹具体对应名称文件

* vis.text 文本
* vis.image 图片
* vis.scatter 2D/3D散点图
* vis.line 线型图
* vis.stem 茎叶图
* vis.heatmap 热力图
* vis.bar 条形图
* vis.histogram 直方图
* vis.boxplot 箱型图
* vis.surf 表面图/立体图
* vis.contour 轮廓图/等高线
* vis.mesh 网格图
* vis.svg SVG图像