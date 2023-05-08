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

