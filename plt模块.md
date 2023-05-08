# plt模块

## 基础

```python
'''
s：形状的大小，默认 20，也可以是个数组，数组每个参数为对应点的大小，数值越大对应的图中的点越大。
c：形状的颜色，"b"：blue   "g"：green    "r"：red   "c"：cyan(蓝绿色，青色)  "m"：magenta(洋红色，品红色) "y"：yellow "k"：black  "w"：white
marker：常见的形状有如下
"."：点                   ","：像素点           "o"：圆形
"v"：朝下三角形   "^"：朝上三角形   "<"：朝左三角形   ">"：朝右三角形
"s"：正方形           "p"：五边星          "*"：星型
"h"：1号六角形     "H"：2号六角形 

"+"：+号标记      "x"：x号标记
"D"：菱形              "d"：小型菱形 
"|"：垂直线形         "_"：水平线形

'''
```

以下两段代码等价

```python
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

```

```python
fig = plt.figure()#首先调用plt.figure()创建了一个**画窗对象fig**
ax = fig.add_subplot(111)#然后再对fix创建默认的坐标区（一行一列一个坐标区）
#这里的（111）相当于（1，1，1），当然，官方有规定，当子区域不超过9个的时候，我们可以这样简写

```

## plt.subplot()

```python
plt.subplot(nrows,ncols,index,**kwargs)
'''
一般我们只用到前三个参数，将整个绘图区域分成 nrows 行和 ncols 列，而index用于对子图进行编号
'''
```

eg:

```python
import matplotlib.pyplot as plt
import numpy as np

#plot 1:
xpoints = np.array([0, 6])
ypoints = np.array([0, 100])

plt.subplot(1, 2, 1)
plt.plot(xpoints,ypoints)
plt.title("plot 1")

#plot 2:
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])

plt.subplot(1, 2, 2)
plt.plot(x,y)
plt.title("plot 2")

plt.suptitle("RUNOOB subplot Test")
plt.show()

```

输出如下：

![subplot_output](D:\Github.blog\ML_notes\datas\subplot_output.png)

## plt.subplots()

```python
matplotlib.pyplot.subplots(nrows=1, ncols=1, *, sharex=False, 
sharey=False, squeeze=True,subplot_kw=None, gridspec_kw=None, **fig_kw)

'''
-nrows：默认为 1，设置图表的行数。
-ncols：默认为 1，设置图表的列数。
-sharex、sharey：设置 x、y 轴是否共享属性
默认为 false，可设置为 ‘none’、‘all’、‘row’ 或 ‘col’。
False 或 none 每个子图的 x 轴或 y 轴都是独立的
True 或 ‘all’：所有子图共享 x 轴或 y 轴
‘row’ 设置每个子图行共享一个 x 轴或 y 轴
‘col’：设置每个子图列共享一个 x 轴或 y 轴。
'''
```

## ax.plot_surface(x,y,z)

用途：生成二维曲面

其中x,y分别是网格化后的数组。

eg:

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x=np.linspace(1,100,10)
y=np.linspace(1,10,20)
x,y=np.meshgrid(x,y)		#网格化

z=x*x+y*y
ax=plt.figure().add_subplot(111,projection='3d')
ax.plot_surface(x,y,z)
plt.show()
```

输出结果：

![plot_surface](D:\Github.blog\ML_notes\datas\plot_surface.png)