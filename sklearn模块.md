# sklearn模块

## model

```python
#model的训练
model.fit(data_X,data_Y)

#model的得分
model.score(test_X,test_Y)

#模型的系数
model.coef_			#以线性模型为例，为斜率
model.intercept		#以线性模型为例，为截距

#输出模型定义的参数
model.get_params()

#输出模型的预测值
model.predict()


```

## train_test_split()

用途：划分数据集

```python
x_train,x_test,y_train,y_test = train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)

'''
*arrays：需要分割的数据集，可以是一个序列、数组、DataFrame等。如果传入的是多个序列或数组，则需要将它们用逗号隔开。

test_size：测试集的大小。默认值为0.25，表示将25%的数据作为测试集。如果test_size和train_size都没有设置，则默认为0.25；如果两个都设置了，则以train_size为准。

train_size：训练集的大小。默认值为0.75，表示将75%的数据作为训练集。

random_state：随机数生成器的种子。可以是一个整数或一个随机数生成器。如果不指定，则每次生成的数据集都不同。

shuffle：在分割之前，是否对数据进行洗牌。默认为True，表示进行洗牌。

stratify：用于分层抽样的标签数组。如果指定了这个参数，则在分割时会根据这个标签数组进行分层抽样，以保证训练集和测试集中的类别比例相同。默认为None，表示不进行分层抽样。

'''
```



eg:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据集
data = pd.read_csv('data.csv')

# 将数据集拆分为特征和目标变量
X = data.drop('target', axis=1)
y = data['target']

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 输出训练集和测试集的形状
print("训练集特征变量形状：", X_train.shape)
print("测试集特征变量形状：", X_test.shape)
print("训练集目标变量形状：", y_train.shape)
print("测试集目标变量形状：", y_test.shape)

```

## classification_report()

用途：用于生成分类模型的评估报告

```python
sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False)
'''
y_true: 真实标签值，一维数组或稀疏矩阵格式。
y_pred: 模型预测值，一维数组或稀疏矩阵格式。
labels: 评估时需要包括的标签列表，即在输出结果中展示的类别，如果为None则默认评估所有类别。
target_names: 每个标签的可读性名称列表，默认为None。(将数字还原字符串表示)
sample_weight: 样本权重数组，与每个样本的预测结果相关联。
digits: 保留的小数位数，默认为2。
output_dict: 如果为True，则输出结果以字典的形式返回。
'''
'''
返回值：

分类模型评估报告，包括各个类别的准确率、召回率、F1-score和支持度等指标。
'''
```

eg:

```python
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=iris.target_names))
'''
output:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.80      0.89        10
   virginica       0.83      1.00      0.91        10

    accuracy                           0.93        30
   macro avg       0.94      0.93      0.93        30
weighted avg       0.94      0.93      0.93        30

'''

```

当使用 `classification_report` 时，它将计算以下几个指标：

- Precision（精确率）：预测为正样本的样本中，真正的正样本占比。其计算公式为：$Precision = \frac{TP}{TP+FP}$，其中 $TP$ 代表真正的正样本数量，$FP$ 代表预测为正样本但实际上是负样本的数量。精确率越高，表示模型在预测正样本时的准确性越高。

- Recall（召回率）：真正的正样本中，被预测为正样本的占比。其计算公式为：$Recall = \frac{TP}{TP+FN}$，其中 $TP$ 代表真正的正样本数量，$FN$ 代表预测为负样本但实际上是正样本的数量。召回率越高，表示模型对正样本的识别能力越强。

- F1-score：精确率和召回率的加权平均值。其计算公式为：$F1-score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$。F1-score 综合了精确率和召回率的指标，用于评估分类模型的综合性能。

- Support（支持度）：每个类别的样本数。在二分类问题中，为真正的正样本和负样本数量之和；在多分类问题中，为每个类别的样本数量之和。

- `accuracy`: 分类准确率，即所有`分类正确`的样本数占 `总样本数` 的比例。

- `macro avg`: 宏平均，对每个类别给出相同的权重，计算出各个类别指标的平均值。

- `weighted avg`: 加权平均，不同类别的样本数量不同，对每个类别的指标进行加权平均，权重为各类别样本数量占总样本数量的比例。 

  在多分类任务中，`macro avg`和`weighted avg`可以更全面地评估模型的性能，`macro avg`主要关注每个类别的分类性能，`weighted avg`则更注重样本量较大的类别的分类性能。

## 交叉验证

### 模型评估

~~~python
from sklearn.model_selection import cross_val_score
'''
`cross_val_score`是scikit-learn库中的一个函数，用于执行交叉验证评估模型性能。它可以帮助我们对机器学习模型进行评估，并提供模型在不同数据子集上的性能指标。
```

参数说明：
- `estimator`：机器学习模型对象，通常是一个实例化的分类器或回归器。
- `X`：特征矩阵，用于训练和评估模型的输入特征。
- `y`：目标向量，用于训练和评估模型的目标变量（可选）。
- `scoring`：性能评估指标，用于衡量模型的性能，默认使用模型的默认评估指标。
- `cv`：交叉验证的策略，可以是一个整数、交叉验证生成器或一个可迭代对象。
- `n_jobs`：并行计算的数量（可选）。

`cross_val_score`函数的工作原理如下：
1. 将数据集拆分为多个子集（折叠）。
2. 针对每个子集，将模型分别进行训练和评估。
3. 返回每个子集上评估指标的结果。

函数返回一个包含每个子集上评估指标结果的数组。通常，我们可以使用平均值或其他汇总统计量来总结这些结果，以获得模型的整体性能指标。

'''
~~~

eg:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 创建一个逻辑回归分类器
classifier = LogisticRegression()

# 执行交叉验证评估
scores = cross_val_score(classifier, X, y, cv=5)

# 打印每个折叠的准确率
print("Cross-validated scores:", scores)

# 打印平均准确率
print("Average accuracy:", scores.mean())
'''
output:
Cross-validated scores: [0.96666667 1.         0.93333333 0.96666667 1.        ]
Average accuracy: 0.9733333333333334

'''
```



### 超参数调优GridSearchCV

网格搜索

```python
from sklearn.model_selection import GridSearchCV
'''
`GridSearchCV`是scikit-learn库中的一个类，用于进行网格搜索（Grid Search）来优化模型的超参数选择。网格搜索是一种通过遍历给定的超参数组合来确定最佳超参数配置的方法。

下面是`GridSearchCV`类的一些重要参数的详细解释：

1. `estimator`：指定要优化的模型估计器（estimator）。它可以是一个回归器（如`LinearRegression`）或分类器（如`SVC`）的实例。

2. `param_grid`：一个字典或列表，用于定义要搜索的超参数空间。字典的键是模型的超参数名称，对应的值是超参数的候选值列表。列表中的元素是一个超参数字典，每个字典定义了一个超参数组合。

3. `scoring`：指定用于评估模型性能的评分指标。它可以是预定义的字符串（如`'accuracy'`，`'f1'`等），也可以是一个自定义的评分函数。

4. `cv`：指定交叉验证的折数（默认值为5）。它确定了将数据划分为多少个训练和验证集的折叠。

5. `refit`：是否在搜索完成后使用最佳参数重新拟合模型（默认值为True）。如果设置为True，则在网格搜索结束后，使用完整训练数据和最佳参数组合对模型进行重新拟合。

6. `n_jobs`：指定并行运行的作业数（默认值为None）。它确定了要使用的并行作业数量，-1表示使用所有可用的CPU核心。

这些是`GridSearchCV`类的一些常用参数。在使用`fit`方法运行网格搜索之后，可以通过访问属性如`best_params_`、`best_score_`和`best_estimator_`来获取最佳参数组合、最佳得分和最佳模型估计器。

网格搜索通过在超参数空间中进行组合的方式，尝试了所有可能的超参数配置，并选择了最佳的超参数组合来优化模型的性能。这使得模型的超参数选择更加准确和自动化。
'''
```

eg:

```python
param_grid = {'C': [0.1, 1, 5, 10, 100],
             'gamma': [0.0005, 0.001, 0.005, 0.01], }
model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
model.fit(x_train_pca, y_train)
print(model.best_estimator_)
print(model.best_params_)

'''
output:
SVC(C=100, class_weight='balanced', gamma=0.005)
{'C': 100, 'gamma': 0.005}
'''
```



## model类型

### LogisticRegression()

```python
from sklearn.linear_model import LogisticRegression
'''
1. `penalty`：正则化项（默认值为'l2'）。它用于控制模型的正则化方式。可选的取值有'l1'（L1正则化）和'l2'（L2正则化）。

2. `C`：正则化强度的倒数（默认值为1.0）。它与正则化项成反比，用于控制模型的正则化力度。较小的C值表示较强的正则化，较大的C值表示较弱的正则化。

3. `solver`：求解器（默认值为'sag'）。它用于指定用于优化问题的算法。常见的求解器包括：'newton-cg'、'lbfgs'、'liblinear'、'sag'和'saga'。不同的求解器适用于不同的问题和数据集大小。

4. `class_weight`：类别权重（默认值为None）。它用于处理不平衡的训练数据。可以设置为'balanced'，使得模型根据类别的频率自动调整权重，更加关注少数类别。

5. `max_iter`：最大迭代次数（默认值为100）。它指定模型的最大迭代次数，用于控制模型的训练时间和收敛性。

6. `multi_class`：多类别分类策略（默认值为'auto'）。它指定多类别问题的处理方式。可选的取值有'auto'、'ovr'和'multinomial'。'auto'表示根据数据自动选择策略，'ovr'表示一对多（One-vs-Rest）策略，'multinomial'表示多项逻辑回归。

7. `random_state`：随机种子（默认值为None）。它用于控制随机数生成的种子，以确保结果的可重复性。

这些只是`LogisticRegression`类的一些常见参数，还有其他一些参数可以进一步调整模型的行为。可以根据具体需求来选择适合的参数值。
'''
```

### LinearRegression()

```python
from sklearn.linear_model import LinearRegression
'''

1. `fit_intercept`：是否拟合截距（默认值为True）。如果设置为True，则模型将拟合截距项，即线性模型中的常数项。如果设置为False，则模型不会拟合截距项。

2. `normalize`：是否对特征进行归一化（默认值为False）。如果设置为True，则模型在训练过程中会对特征进行归一化处理，即将特征按列进行减均值、除以标准差的操作。

3. `copy_X`：是否复制输入数据（默认值为True）。如果设置为True，则在模型训练期间会对输入数据进行复制。如果设置为False，则直接使用原始输入数据，可能会在原始数据上进行修改。

4. `n_jobs`：并行计算的数量（默认值为1）。它指定了模型在训练过程中使用的并行计算的数量。如果设置为-1，则使用所有可用的CPU核心进行并行计算。

5. `positive`：是否限制预测值为非负数（默认值为False）。如果设置为True，则模型会对预测值进行非负约束。


'''
```

### SVC()

用途：SVM分类器

```python
from sklearn.svm import SVC
'''
参数解释：
1. `C`：正则化参数（默认值为1.0）。它控制了错误分类样本对于决策边界的影响程度。较小的C值使模型更容易容忍错误分类，而较大的C值使模型更加关注正确分类。

2. `kernel`：核函数类型（默认值为'rbf'）。核函数定义了样本间的相似度度量，用于构建决策边界。常见的核函数有线性核（'linear'）、多项式核（'poly'）、径向基函数（'rbf'）等。

3. `degree`：多项式核函数的阶数（默认值为3）。它只有在核函数为多项式核（'poly'）时才会起作用。

4. `gamma`：核函数的系数（默认值为'auto'）。它影响了样本与支持向量之间的相似度计算。较小的gamma值意味着相似度下降得更快，决策边界更加平滑；较大的gamma值意味着相似度下降得更慢，决策边界更加关注支持向量。

5. `class_weight`：类别权重（默认值为None）。它用于处理不平衡的训练数据。可以设置为'balanced'，使得模型根据类别的频率自动调整权重，更加关注少数类别。

6. `probability`：是否启用概率估计（默认值为False）。当设置为True时，模型会计算类别的概率预测。

7. `shrinking`：是否启用启发式收缩（默认值为True）。启发式收缩可以加快模型的训练速度，但可能会牺牲一些精确度。

8. `tol`：停止训练的容忍度（默认值为1e-3）。它指定了模型停止迭代的容忍度，即当目标函数的变化小于容忍度时停止训练。

这些只是`SVC`类的一些常见参数，还有其他一些参数可以进一步调整模型的行为。可以根据具体需求来选择适合的参数值。
'''
```

### PCA()

用途：对数据进行降维

```python
from sklearn.decomposition import PCA

'''
1. `n_components`：保留的主成分个数或保留的方差比例（默认值为None）。如果指定一个整数值，则保留对应数量的主成分。如果指定一个0到1之间的浮点数，则保留足够数量的主成分以解释指定的方差比例。

2. `whiten`：是否对数据进行白化（默认值为False）。如果设置为True，则对降维后的数据进行白化处理，即使得特征之间的协方差为零且每个特征的方差为1。

3. `svd_solver`：奇异值分解（Singular Value Decomposition，SVD）求解器的选择（默认值为'auto'）。可选的取值有'auto'、'full'、'arpack'和'randomized'。它决定了PCA算法在进行奇异值分解时使用的求解方法。

4. `iterated_power`：奇异值分解迭代次数（默认值为'auto'）。当`svd_solver`为'randomized'时，它指定随机化SVD算法的迭代次数。

5. `random_state`：随机种子（默认值为None）。它用于控制随机数生成的种子，以确保结果的可重复性。在某些求解器中，随机性被引入以提高算法的效率。

'''
```

eg:

```python
x_train,x_test,y_train,y_test=train_test_split(lfw_people.data,lfw_people.target)

n_components=100	#保留一百个特征

#定义模型
pca=PCA(n_components=n_components,whiten=True)
pca.fit(lfw_people.data)
#转换训练集，测试集
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
```

### TfidfVectorizer()和CountVectorizer(）

`TfidfVectorizer()`和`CountVectorizer()`是scikit-learn库中用于文本特征提取的两个常用类。它们之间有一些关键的区别：

1. 特征表示方法：
   - `CountVectorizer()`：将文本转换为词频向量表示，即每个文档中每个单词的出现次数。
   - `TfidfVectorizer()`：将文本转换为TF-IDF（Term Frequency-Inverse Document Frequency）向量表示，考虑了单词在文档集合中的重要性。

2. 特征值计算：
   - `CountVectorizer()`：计算的是每个单词在文档中的出现次数。
   - `TfidfVectorizer()`：计算的是每个单词在文档中的TF-IDF值，结合了单词的频率和在文档集合中的重要性。

3. 特征矩阵稀疏性：
   - `CountVectorizer()`：生成的特征矩阵通常是稀疏矩阵，因为文档中的大多数单词都不会同时出现。
   - `TfidfVectorizer()`：生成的特征矩阵也通常是稀疏矩阵，因为大多数单词在文档集合中出现的次数相对较少。

4. 特征权重计算：
   - `CountVectorizer()`：没有考虑单词在文档集合中的重要性，仅使用单词的频率作为特征权重。
   - `TfidfVectorizer()`：考虑了单词的频率和在文档集合中的重要性，使用TF-IDF值作为特征权重。

5. 停用词过滤：
   - `CountVectorizer()`和`TfidfVectorizer()`都支持停用词过滤，可以通过设置`stop_words`参数来指定停用词列表。

选择使用`CountVectorizer()`还是`TfidfVectorizer()`取决于具体的任务和数据集。一般来说，如果文本中的常用单词对于任务并不重要，那么`TfidfVectorizer()`可能更适合，因为它可以将重点放在罕见但有信息量的单词上。而`CountVectorizer()`则更适合简单的文本表示或某些特定任务，如词频分析。

#### CountVectorizer()

用途将文本数据转换为特征向量表示

```python
from sklearn.feature_extraction.text import CountVectorizer

'''
`CountVectorizer()`是scikit-learn库中的一个类，用于将文本数据转换为特征向量表示。它执行以下几个主要步骤：

1. 文本分词：`CountVectorizer()`将输入的文本数据拆分为单个单词（或称为标记）。默认情况下，它使用空格作为分隔符来拆分文本，但也可以通过正则表达式进行自定义分隔。

2. 构建词汇表：它创建一个词汇表，包含从输入文本中提取的所有不重复的单词。每个单词都被分配一个唯一的整数索引。

3. 计算词频：对于每个文本样本，`CountVectorizer()`计算每个单词在该样本中的出现次数。这形成了文本样本的特征向量。

4. 向量化：最后，`CountVectorizer()`将每个文本样本转换为一个特征向量，其中每个特征表示词汇表中的一个单词，特征值表示该单词在文本样本中的出现次数。生成的特征向量是一个稀疏向量，其中大多数元素为零，因为大部分文本样本中的单词并非全部出现。



`CountVectorizer()`还有一些可选参数，可以用于自定义向量化的过程，例如：

- `stop_words`：指定停用词列表，用于去除常见但无实际意义的单词。
- `ngram_range`：指定要考虑的n-gram范围，以捕捉多个连续单词的组合。
- `max_features`：指定要保留的最大特征数。
- 其他参数可以查阅scikit-learn官方文档以了解更多详细信息。


'''
```



eg:

```python
from sklearn.feature_extraction.text import CountVectorizer

# 创建一个CountVectorizer对象
cv = CountVectorizer()

# 文本数据
text_data = ["I love to eat pizza.",
             "I love to eat burgers."]

# 将文本数据进行向量化
feature_vector = cv.fit_transform(text_data)

# 打印特征向量表示
print(feature_vector.toarray())
'''
output:
[[1 1 1 1 0]
 [1 1 1 0 1]]

'''
```

#### TfidfVectorizer()

TF-IDF（Term Frequency-Inverse Document Frequency）是一种用于衡量一个词对于文档集合中每个文档的重要性的统计方法。它结合了两个关键因素：词频（Term Frequency，TF）和逆文档频率（Inverse Document Frequency，IDF）。

TF（词频）指的是一个词在文档中出现的频率。它通过计算一个词在文档中出现的次数来衡量其重要性。一般来说，一个词在文档中出现的次数越多，它对于该文档的重要性也越高。

IDF（逆文档频率）是对一个词在文档集合中的普遍重要性的度量。它通过计算一个词在文档集合中出现的文档数的倒数来衡量。如果一个词在大多数文档中都出现，它对于区分不同文档的能力就较低，因此其IDF值较低。相反，如果一个词只在少数文档中出现，它对于区分文档的能力较高，因此其IDF值较高。

TF-IDF通过将TF和IDF结合起来计算一个词的权重，以反映其在文档集合中的重要性。具体而言，TF-IDF值是通过将一个词的TF乘以其对应的IDF来计算得到的。

TF-IDF的计算公式为：

TF-IDF = TF * IDF

在实际应用中，通常还会对TF-IDF值进行归一化或进行其他的变换操作，以便更好地适应具体任务的需求。

TF-IDF在文本挖掘、信息检索和自然语言处理等领域中广泛应用。它能够帮助识别文本中重要的关键词，过滤掉常见的无意义词语，并在文本之间计算相似性。通过使用TF-IDF，可以提高文本特征的表达能力，并对不同文档之间的重要性和相似度进行建模和分析。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
'''
`TfidfVectorizer()`是scikit-learn库中的一个类，用于将文本数据转换为TF-IDF特征表示。TF-IDF（Term Frequency-Inverse Document Frequency）是一种常用的文本特征表示方法，用于衡量一个词对于文档集合中每个文档的重要性。

`TfidfVectorizer()`的主要作用是将原始文本数据转换为稀疏矩阵表示，其中每个文档由一个向量表示，向量的每个元素表示一个单词在该文档中的TF-IDF值。

下面是`TfidfVectorizer()`的一些重要参数和用法：

- `stop_words`（默认为None）：停用词列表，用于过滤常见的无意义单词。

- `tokenizer`（默认为None）：用于分词的函数或可调用对象，可以自定义分词逻辑。

- `max_df`（默认为1.0）：过滤掉高于给定阈值的文档频率的单词。

- `min_df`（默认为1）：过滤掉低于给定阈值的文档频率的单词。

- `ngram_range`（默认为(1, 1)）：表示要提取的n-gram的范围。例如，(1, 1)表示只提取单个词，(1, 2)表示提取单个词和二元词组。

- `fit_transform(raw_documents, y=None)`：用于训练并将原始文本数据转换为TF-IDF特征表示的方法。`raw_documents`表示原始文本数据，`y`表示可选的目标向量。

- `get_feature_names_out()`：返回生成的特征向量中的单词列表。

'''
```

eg:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 原始文本数据
corpus = ["I love pizza.",
          "I hate burgers.",
          "I enjoy pasta."]

# 创建一个TfidfVectorizer对象
tfidf_vectorizer = TfidfVectorizer()

# 将文本数据转换为TF-IDF特征表示
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# 打印特征矩阵
print(tfidf_matrix.toarray())

# 打印特征词汇表
feature_names = tfidf_vectorizer.get_feature_names_out()
print(feature_names)

'''
output:
[[0.         0.         0.         0.70710678 0.         0.70710678]
 [0.70710678 0.         0.70710678 0.         0.         0.        ]
 [0.         0.70710678 0.         0.         0.70710678 0.        ]]
['burgers' 'enjoy' 'hate' 'love' 'pasta' 'pizza']
'''

```

### MultinomialNB()

用途：朴素贝叶斯分类器的多项式模型

```python
from sklearn.naive_bayes import MultinomialNB
'''
`MultinomialNB`类的主要参数包括：

1. `alpha`（默认为1.0）：平滑参数，用于处理在训练数据中未见过的特征。较小的`alpha`值表示较强的平滑，可以减少特征的过拟合，但可能会损失一些训练数据的细节。较大的`alpha`值表示较弱的平滑，模型更倾向于将训练数据中观察到的特征频率直接用于预测，但可能对未见过的特征过于敏感。

2. `fit_prior`（默认为True）：指定是否学习类别的先验概率。如果设置为True，则会根据训练数据中的类别频率自动学习先验概率。如果设置为False，则会使用统一的先验概率。

3. `class_prior`（默认为None）：手动指定类别的先验概率。如果`fit_prior`为False且`class_prior`为None，则将使用统一的先验概率。如果指定了`class_prior`，则会使用给定的先验概率。

4. `fit(X, y, sample_weight=None)`：用于训练分类器模型的方法。其中，`X`是特征矩阵，`y`是目标向量，`sample_weight`是样本权重。`X`表示训练样本的特征向量，`y`表示训练样本的类标签。`sample_weight`用于指定每个样本的权重，可用于调整不平衡数据集中的样本重要性。

5. `predict(X)`：用于对新的样本进行分类预测的方法。其中，`X`是特征矩阵。该方法返回预测的类标签。

6. `predict_proba(X)`：用于返回预测样本属于每个类别的概率的方法。其中，`X`是特征矩阵。返回的结果是一个二维数组，每行对应一个样本，每列对应一个类别，表示样本属于每个类别的概率。


需要注意的是，朴素贝叶斯分类器的性能受参数选择的影响较小，因此通常情况下，使用默认参数即可获得较好的结果。然而，在特定问题和数据集上，根据实际情况进行参数调整和优化可能会有所帮助。

'''
```

eg:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 创建一个CountVectorizer对象
cv = CountVectorizer()

# 文本数据
text_data = ["I love pizza.",
             "I hate burgers.",
             "I enjoy pasta."]

# 将文本数据向量化
feature_vector = cv.fit_transform(text_data)

# 创建一个MultinomialNB分类器
classifier = MultinomialNB()

# 训练分类器
classifier.fit(feature_vector, ['positive', 'negative', 'neutral'])

# 测试数据
test_data = ["I like pizza and burgers."]

# 将测试数据向量化
test_vector = cv.transform(test_data)

# 预测类别
predicted_class = classifier.predict(test_vector)

# 打印预测结果
print(predicted_class)
'''
output:
['positive']

'''
```

