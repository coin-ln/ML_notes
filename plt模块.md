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

```python
plt.rcParams['figure.facecolor'] = 'white'	#白色背景
plt.rcParams['font.family'] = 'SimHei'  # 将全局字体设置为中文字体
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

## plt.xticks()||plt.yticks()

用途：设置坐标轴

```python
'''
`plt.xticks()`是Matplotlib库中的一个函数，用于设置当前坐标轴的x轴刻度标签。它允许你自定义刻度标签的位置和文本。

下面是对`plt.xticks()`函数的解释：

1. 参数`ticks`：要在x轴上显示的刻度标签的位置。它可以是一个数字列表，表示刻度标签所在的位置。例如，`ticks = [0, 1, 2, 3]`将在x轴上的位置0、1、2和3处显示刻度标签。如果未提供此参数，默认情况下将使用自动计算的刻度标签。

2. 参数`labels`：要显示的刻度标签的文本列表。它可以是一个字符串列表，包含与刻度标签位置对应的文本。例如，`labels = ['A', 'B', 'C', 'D']`将在刻度标签的位置0、1、2和3处显示文本标签'A'、'B'、'C'和'D'。

通过使用`plt.xticks()`函数，你可以自定义x轴上的刻度标签。可以指定刻度标签的位置和文本，以便更好地展示数据和增强可读性。
'''
```

eg:

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.xticks([1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E'])
plt.show()
```

output:

![plt.xticks](D:\Github.blog\ML_notes\datas\plt.xticks.png)

tips:空坐标轴用法

```python
plt.xticks(())
plt.yticks(())
```



## plt.xlim||plt.ylim

用途：规定坐标轴范围

```python
import matplotlib.pyplot as plt

# 假设 x 和 y 是数据集
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 绘制散点图
plt.scatter(x, y)

# 设置坐标轴范围
plt.xlim(0, 10)
plt.ylim(0, 20)

# 显示图形
plt.show()

```

## plt.subplots_adjust()

用途：调整子画之间的布局和间距

```python
`plt.subplots_adjust`是Matplotlib库中的一个函数，用于调整子图的布局和间距。它允许对子图之间的空白区域进行调整，以便更好地适应图形的显示。

下面是对`plt.subplots_adjust`函数中参数的详细解释：

1. `bottom`：底部边界的位置（默认值为0）。它表示子图的底部边界与整个图形区域底部之间的相对位置。

2. `left`：左侧边界的位置（默认值为0.125）。它表示子图的左侧边界与整个图形区域左侧之间的相对位置。

3. `right`：右侧边界的位置（默认值为0.9）。它表示子图的右侧边界与整个图形区域右侧之间的相对位置。

4. `top`：顶部边界的位置（默认值为0.9）。它表示子图的顶部边界与整个图形区域顶部之间的相对位置。

5. `hspace`：水平间距（默认值为0.2）。它表示子图之间的水平间距，即子图之间的空白区域的大小。

通过调整这些参数的值，可以控制子图在图形区域中的位置和间距，以实现更好的布局和可视化效果。

在你提供的代码中，`plt.subplots_adjust`被用于调整子图的布局和间距。具体而言，它将底部边界设置为0，左侧边界设置为0.01，右侧边界设置为0.99，顶部边界设置为0.9，水平间距设置为0.35。这样的调整可以使子图更好地适应整个图形区域，并提供适当的间距以增强可读性和美观性。
```

eg:

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(2*x)
y4 = np.cos(2*x)

# 创建子图
fig, axes = plt.subplots(2, 2)

# 绘制子图
axes[0, 0].plot(x, y1)
axes[0, 0].set_title('Plot 1')
axes[0, 1].plot(x, y2)
axes[0, 1].set_title('Plot 2')
axes[1, 0].plot(x, y3)
axes[1, 0].set_title('Plot 3')
axes[1, 1].plot(x, y4)
axes[1, 1].set_title('Plot 4')

# 调整布局和间距
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, hspace=0.4, wspace=0.4)

# 展示图形
plt.show()

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