import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

iris_data_set = pd.read_csv('E:\\PyCharm-Workspace\\DataAnalysis\\data\\03 Iris\\DecisionTreeClassifier\\iris.csv')

# x是4列特征
x = iris_data_set.iloc[:, 0:4].values
# y是1列标签
y = iris_data_set.iloc[:, -1].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# 利用GridSearchCV选择最优参数
model = DecisionTreeClassifier()
param = {'criterion': ['gini', 'entropy'], 'max_depth': [30, 50, 60, 100], 'min_samples_leaf': [2, 3, 5, 10], 'min_impurity_decrease': [0.1, 0.2, 0.5]}
grid = GridSearchCV(model, param_grid=param, cv=5)
grid.fit(x_train, y_train)
print('最优分类器:', grid.best_estimator_)
print('最优超参数：', grid.best_params_)
print('最优分数:', grid.best_score_)

# 利用决策树分类器构建分类模型
model = grid.best_estimator_
y_pre = model.predict(x_test)

print('正确标签：', y_test)
print('预测结果：', y_pre)

print('训练集分数：', model.score(x_train, y_train))
print('测试集分数：', model.score(x_test, y_test))

# 混淆矩阵
conf_mat = confusion_matrix(y_test, y_pre)
print('混淆矩阵：')
print(conf_mat)

# 分类指标文本报告（精确率、召回率、F1值等）
print('分类指标报告：')
print(classification_report(y_test, y_pre))

# 特征重要性
print(model.feature_importances_)

# 画图展示训练结果
fig = plt.figure()
ax = fig.add_subplot(111)
f1 = ax.scatter(list(range(len(x_test))), y_test, marker='*')
f2 = ax.scatter(list(range(len(x_test))), y_pre, marker='o')
plt.legend(handles=[f1, f2], labels=['True', 'Prediction'])
plt.show()
