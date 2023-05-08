# numpy模块

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

