import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

data = pd.read_csv('D:\\iris.csv', sep=',', encoding='utf-8')
x = data[['A', 'B', 'C', 'D']].as_matrix()
real = data['E'].as_matrix()

# 第1种实现：KMeans算法
# kms = KMeans(n_clusters=3)  # 传入要分类的数目
# y = kms.fit_predict(x)

# 第2种实现：DBSCAN算法
dbscan = DBSCAN(eps=0.5, min_samples=13)
dbscan.fit(x)
y = dbscan.labels_

count = 0
for i in range(len(real)):
    if abs(int(y[i])) == abs(int(real[i])):
        count = count + 1
print('正确：' + str(count))
acc = round(count / len(real), 4) * 100
print('正确率：' + str(acc) + '%')
