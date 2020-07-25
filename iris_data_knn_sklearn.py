import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 读取数据
iris_data_set = pd.read_csv("D:\\Michael\\Documents\\01 项目\\01 大数据平台\\14 数据分析\\02 油压减振器故障分类\\04 测试数据\\鸢尾花\\iris.csv")
# x是4列特征
x = iris_data_set.iloc[:, 0:4].values
# y是1列标签
y = iris_data_set.iloc[:, -1].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 将特征转为一维数组
y_train = y_train.flatten()
y_test = y_test.flatten()

# 建模、训练、预测
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
y_pre = knn_model.predict(x_test)

print('正确标签：', y_test)
print('预测结果：', y_pre)

# 混淆矩阵
conf_mat = confusion_matrix(y_test, y_pre)
print('混淆矩阵：')
print(conf_mat)

# 分类指标文本报告（精确率、召回率、F1值等）
print('分类指标报告：')
print(classification_report(y_test, y_pre))
