# seaborn库

Seaborn是一个基于matplotlib的Python可视化库，专门用于绘制统计图表和数据可视化。Seaborn的设计理念是提供一种高度可定制和美观的界面，可以让用户更容易地探索和理解数据。Seaborn提供了许多高级数据可视化工具和函数，可以方便地创建各种类型的图表，如热力图、散点图、条形图、密度图等等。同时，Seaborn还内置了一些数据集，方便用户快速绘制可视化图表。Seaborn的API简单易用，通过简单的几行代码就能快速生成美观的图表。Seaborn也与pandas数据框紧密结合，可以直接使用pandas数据框作为输入。

```python
import seaborn as sns
```



## pairplot()

按参数hue分类，然后生成其他任意两变量之间的关系。

```python
sns.pairplot(data, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='auto', markers=None, height=2.5, aspect=1, corner=False, dropna=True, plot_kws=None, diag_kws=None, grid_kws=None, size=None)
'''
data: 需要绘制的数据集，可以是DataFrame或Numpy数组类型。
hue: 指定一个分类变量，用于按照不同的类别对数据进行分类，然后用不同颜色绘制散点图。
vars: 指定要绘制的变量的列表。如果不设置，则默认使用数据集中所有的数值型变量。
x_vars和y_vars: 分别指定要绘制的自变量和因变量的列表。
kind: 指定要绘制的图形类型，可选参数包括scatter（散点图）、reg（回归图）、hist（直方图）、kde（核密度图）、hex（六角箱图）等等。
diag_kind: 指定对角线上的图形类型，可选参数包括auto（自动判断）、hist（直方图）、kde（核密度图）等等。
markers: 指定散点图的标记样式，可以是单个标记，也可以是一个列表。
height和aspect: 指定子图的高度和宽高比。：
'''
```

eg:

```python
import seaborn as sns
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species")
```

输出如下：

![sns.pairplot](D:\Github.blog\ML_notes\datas\sns.pairplot.png)

## heatmap()

用途:热力图

```python
sns.heat(data,cmap,annot,fmt,linewidths,square,cbar,cbar_kws)

'''
data：要绘制的矩阵数据。
cmap：颜色映射，即将数据映射到颜色的规则，例如Reds、Blues、Greens、Oranges等。
annot：是否在热力图上显示数值。
fmt：注释中显示的数字格式。
linewidths：热力图中每个单元格边框的线宽。
square：是否将每个单元格调整为正方形。
cbar：是否在热力图侧面显示颜色条。
cbar_kws：颜色条的一些参数，如方向、位置、标签等。
'''
```

eg:

```python
import seaborn as sns
import numpy as np

# 生成一个5x5的随机矩阵
data = np.random.rand(5, 5)

# 画热力图
sns.heatmap(data, cmap='Blues', annot=True, fmt=".2f", linewidths=.5)

```

![sns.heatmap](D:\Github.blog\ML_notes\datas\sns.heatmap.png)

## scatterplot()

```python
scatterplot()
'''
x, y：必需参数，表示要绘制散点图的变量名。x、y 必须是数据框中的列名或者向量。

data：必需参数，表示要绘制散点图的数据框。

hue：用于根据数据框中的分类变量对散点图进行着色。可以传递分类变量的名称或者数据框中分类变量的列名。如果传递了此参数，则每个分类的数据将使用不同的颜色进行绘制。

style：用于根据数据框中的另一个分类变量对散点图进行样式化。可以传递分类变量的名称或者数据框中分类变量的列名。如果传递了此参数，则每个分类的数据将使用不同的标记进行绘制。

size：用于根据另一个连续变量调整散点的大小。可以传递连续变量的名称或者数据框中连续变量的列名。如果传递了此参数，则每个点的大小将根据指定的变量进行调整。

sizes：用于自定义散点的大小范围。可以传递一个列表或元组，包含两个数字，分别表示最小和最大大小。如果指定了该参数，将会忽略 size 参数。

alpha：用于设置散点图的透明度。取值范围为 0 到 1，0 表示完全透明，1 表示完全不透明。

palette：用于设置分类变量的调色板。可以传递一个 Seaborn 调色板名称、matplotlib 颜色映射名称或一个颜色列表。

legend：用于指定是否显示图例。

markers：用于自定义标记的样式。可以传递标记的名称、标记的列表、标记的字典或一个函数，根据输入数据动态地生成标记。

ax：用于指定要绘制图形的坐标轴。如果没有指定，则默认使用当前的坐标轴。

ci：用于指定绘制误差线的置信区间大小。

err_style：用于设置误差线的样式。

x_jitter、y_jitter：用于添加抖动来解决数据的重叠问题。

alpha：用于设置散点图的透明度。取值范围为 0 到 1，0 表示完全透明，1 表示完全不透明。

'''
```

