# numpy模块

## 样本统计值

### mean()

用途：平均值

```python
np.mean(a,axis=None,dtype=None,keepdims=False)
'''
a:数组
axis: 轴，对指定轴进行计算，默认对所有元素进行计算。
dtype: 返回结果的数据类型，默认为None，表示使用数组中元素的数据类型。
keepdims: 是否保留维度，如果为True，则返回的数组与原数组的维度相同。
'''
```

### var()

用途：方差

```python
np.var(a, axis=None, dtype=None, ddof=0, keepdims=False)
'''
a: 数组。
axis: 轴，对指定轴进行计算，默认对所有元素进行计算。
dtype: 返回结果的数据类型，默认为None，表示使用数组中元素的数据类型。
ddof: 自由度，用于计算无偏方差，默认为0，表示使用全体数据的方差。
keepdims: 是否保留维度，如果为True，则返回的数组与原数组的维度相同。
'''
```

### std()

用途：标准差

```python
np.std(a, axis=None, dtype=None, ddof=0, keepdims=False)
'''
a: 数组。
axis: 轴，对指定轴进行计算，默认对所有元素进行计算。
dtype: 返回结果的数据类型，默认为None，表示使用数组中元素的数据类型。
ddof: 自由度，用于计算无偏标准差，默认为0，表示使用全体数据的标准差。
keepdims: 是否保留维度，如果为True，则返回的数组与原数组的维度相同。
'''
```

### cov()

用途：协方差

只能计算2维以内

```python
numpy.cov(a, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)
'''
a: 数组。
y: 可选，数组。如果指定，则与m一起计算协方差矩阵。
rowvar: 可选，布尔值。如果为True，则将每一行看作一个变量，每一列看作一个观测值，即对行进行计算协方差矩阵；如果为False，则将每一列看作一个变量，每一行看作一个观测值，即对列进行计算协方差矩阵。
bias: 可选，布尔值。如果为True，则偏置为N，否则为N-1。
ddof: 可选，整数。自由度，用于计算无偏协方差矩
返回的矩阵是变量与变量之间的协方差，如C[i，j]描述的是第i个变量与第j个变量之间的协方差
'''
```

即：

rowvar=True时返回n×n矩阵（假设初始矩阵为n×m)，即每行是一个特征变量

rowvar=False时m*m的矩阵 即每列是一个特征变量

eg:

```python
import numpy as np
a=np.array([1,23,4,6,21,6,34,1,32,23]).reshape(2,5)
b=np.cov(a,rowvar=0)
c=np.cov(a,rowvar=1)
a,b,c
'''
output:
(array([[ 1, 23,  4,  6, 21],
        [ 6, 34,  1, 32, 23]]),
 array([[ 12.5,  27.5,  -7.5,  65. ,   5. ],
        [ 27.5,  60.5, -16.5, 143. ,  11. ],
        [ -7.5, -16.5,   4.5, -39. ,  -3. ],
        [ 65. , 143. , -39. , 338. ,  26. ],
        [  5. ,  11. ,  -3. ,  26. ,   2. ]]),
 array([[104.5 , 102.75],
        [102.75, 225.7 ]]))
'''
```



## meshgrid()

用途：生成网格数组，应用于作图

eg:

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.array([0, 0.5, 1])
y = np.array([0,1])

xv,yv = np.meshgrid(x, y)
print(xv,yv)
plt.plot(xv,yv)
plt.grid(True)
plt.show()
'''
xv：生成len(y)组x
yv：生成len(y)组元素，每组元素 全部 等于y[i]
'''


'''
output
[[0.  0.5 1. ]
 [0.  0.5 1. ]] [[0 0 0]
 [1 1 1]]
'''
```

```python
plt.grid(False)
```

![meshgrid_false](D:\Github.blog\ML_notes\datas\meshgrid_false.png)

```python
plt.grid(True)
```

![meshgrid_true](D:\Github.blog\ML_notes\datas\meshgrid_true.png)

## linspace()||arrange()

linspace(s,t,nums)

arrange(s,t,steps)

```python
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,10,2)
y=np.arange(0,6,2)
x,y
'''
output
(array([ 0., 10.]), array([0, 2, 4]))
'''

'''
相同点：前两个参数分别为起始点和结束点
不同点：
linspace [s,t]，第三个参数表示数量
arrange [s,t)，第三个参数表示步长
'''

```

## concatenate()

用途：拼接数组

```python
np.concatenate((a1,a2,a3....),axis,out)
'''
a1,a2,a3...:为要拼接的数组
axis:表示沿哪个轴拼接，即该轴shape改变
out:指定数组输出的形状，必须与连接后的数组shape相同
'''
```

eg：

```python
#axis参数
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.concatenate((a, b), axis=1)
print(c)  # [[1 2 5 6]
           #  [3 4 7 8]]
           #（2,2)->(2,4)

```

```python
#out参数
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.zeros(6)
np.concatenate((a, b), out=c)
print(c)  # [1. 2. 3. 4. 5. 6.]

```

## linalg

说明：numpy中线性代数模块

下面是一些常用的 `linalg` 函数：

- `det()`：计算矩阵的行列式。
- `inv()`：计算矩阵的逆矩阵。
- `solve()`：求解线性方程组 Ax = b。
- `eig()`：计算矩阵的特征值和特征向量。
- `svd()`：计算矩阵的奇异值分解。
- `norm()`：计算矩阵或向量的范数。

```python
import numpy as np

# 创建一个 3x3 的矩阵
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([1, 2, 3])

det_A = np.linalg.det(A)
inv_A = np.linalg.inv(A)
x = np.linalg.solve(A, b)
eig_vals, eig_vecs = np.linalg.eig(A)

# 输出结果
print(det_A)  #det():-9.51619735392994e-16
print(inv_A)
print(x)	#solve():[-0.23333333  0.46666667  0.1       ]

print(eig_vals)  # [ 1.61168440e+01 -1.11684397e+00 -1.30367773e-15]
print(eig_vecs)  # [[-0.23197069 -0.78583024  0.40824829]
                 #  [-0.52532209 -0.08675134 -0.81649658]
                 #  [-0.8186735   0.61232756  0.40824829]]

'''

inv():
[[-0.94444444  0.44444444  0.05555556]
 [ 0.44444444 -0.11111111  0.22222222]
 [ 0.05555556  0.22222222 -0.11111111]]
 
'''


```

```python
import numpy as np

# 创建一个 3x3 的矩阵 A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 使用 svd() 函数对 A 进行奇异值分解
U, S, VT = np.linalg.svd(A)

print("U =\n", U)
print("S =\n", S)
print("VT =\n", VT)

'''
U =
[[-0.21483724  0.88723069 -0.40824829]
 [-0.52058739  0.24964395  0.81649658]
 [-0.82633754 -0.38794279 -0.40824829]]
S =
[1.68481034e+01 1.06836951e+00 3.33475287e-16]
VT =
[[-0.47967118 -0.57236779 -0.66506439]
 [-0.77669099 -0.07568647  0.62531805]
 [ 0.40824829 -0.81649658  0.40824829]]

'''
```

## sort()

用途：排序

**返回排序后的数组，不改变原数组**

```python
numpy.sort(a, axis=-1, kind=None, order=None)
'''
a:数组
axis:沿哪个轴排序,默认最后一轴
kind:排序方式，包括'quicksort'、'mergesort'和'heapsort'三		种，默认为'quicksort'
order:按照哪个字段排序
'''
```

eg:

```python
import numpy as np

a = np.array([3, 1, 4, 2])
sorted_a = np.sort(a)
print(sorted_a)  # 输出 [1 2 3 4]

a = np.array([[3, 1], [4, 2]])
sorted_a = np.sort(a, axis=None)
print(sorted_a)  # 输出 [1 2 3 4]
```

## argsort()

用途：返回排过序后的索引数组

**返回排序后的数组索引，不改变原数组索引**

```python
numpy.argsort(a, axis=-1, kind='quicksort', order=None)
'''
参数同sort一样
'''
```

eg:

```python
import numpy as np

a = np.array([3, 1, 4, 2])
idx = np.argsort(a)
print(idx)  # 输出 [1 3 0 2]
```

