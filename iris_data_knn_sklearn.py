import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# 读取数据
iris_data_set = pd.read_csv('E:\\PyCharm-Workspace\\DataAnalysis\\data\\03 Iris\\DecisionTreeClassifier\\iris.csv')
# x是4列特征
x = iris_data_set.iloc[:, 0:4].values
# y是1列标签
y = iris_data_set.iloc[:, -1].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# 将特征转为一维数组
y_train = y_train.flatten()
y_test = y_test.flatten()

# 利用GridSearchCV选择最优参数
knn_model = KNeighborsClassifier()
param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 20)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 20)],
        'p':[i for i in range(1, 6)]
    }
]
grid = GridSearchCV(knn_model, param_grid=param_grid, cv=5)
grid.fit(x_train, y_train)
print('最优分类器:', grid.best_estimator_)
print('最优超参数：', grid.best_params_)
print('最优分数:', grid.best_score_)

# 预测
knn_model = grid.best_estimator_
y_pre = knn_model.predict(x_test)

print('正确标签：', y_test)
print('预测结果：', y_pre)

print('训练集分数：', knn_model.score(x_train, y_train))
print('测试集分数：', knn_model.score(x_test, y_test))

# 混淆矩阵
conf_mat = confusion_matrix(y_test, y_pre)
print('混淆矩阵：')
print(conf_mat)

# 分类指标文本报告（精确率、召回率、F1值等）
print('分类指标报告：')
print(classification_report(y_test, y_pre))

# 画图展示训练结果
fig = plt.figure()
ax = fig.add_subplot(111)
f1 = ax.scatter(list(range(len(x_test))), y_test, marker='*')
f2 = ax.scatter(list(range(len(x_test))), y_pre, marker='o')
plt.legend(handles=[f1, f2], labels=['True', 'Prediction'])
plt.show()
