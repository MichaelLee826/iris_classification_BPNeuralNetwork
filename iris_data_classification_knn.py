import csv
import random
import numpy as np
import operator
import datetime


def open_file(file_name):
    """
    打开数据集，进行数据处理
    :param file_name: 数据集的路径
    :return: 返回数据集的 特征、标签、标签名
    """
    with open(file_name) as csv_file:
        data_file = csv.reader(csv_file)

        # temp读取的是csv文件的第一行，相当于表头
        temp = next(data_file)

        # 数据集中数据的总数量
        n_samples = int(temp[0])

        # 数据集中特征值的种类个数
        n_features = int(temp[1])

        # 标签名
        labels_names = np.array(temp[2:])

        # 特征集，行数为数据集数量，列数为特征值的种类个数
        features = np.empty((n_samples, n_features))

        # 标签集，行数为数据集数量，1列，数据格式为int
        labels = np.empty((n_samples,), dtype=np.int)

        for i, j in enumerate(data_file):
            # 将数据集中的将数据转化为矩阵，数据格式为float
            # 将数据中从第一列到倒数第二列中的数据保存在data中
            features[i] = np.asarray(j[:-1], dtype=np.float64)

            # 将数据集中的将数据转化为矩阵，数据格式为int
            # 将数据集中倒数第一列中的数据保存在target中
            labels[i] = np.asarray(j[-1], dtype=np.int)

    # 返回 数据，标签 和标签名
    return features, labels, labels_names


def random_number(data_size):
    """
    该函数使用shuffle()打乱一个包含从0到数据集大小的整数列表。因此每次运行程序划分不同，导致结果不同
    :param data_size: 数据集大小
    :return: 返回一个列表
    """
    number_set = []
    for i in range(data_size):
        number_set.append(i)

    random.shuffle(number_set)

    return number_set


def split_data_set(features_set, labels_set, rate=0.20):
    """
    分割数据集，默认数据集的25%是测试集
    :param features_set: 数据集
    :param labels_set: 标签数据
    :param rate: 测试集所占的比率
    :return: 返回训练集数据、训练集标签、测试集数据、测试集标签
    """
    # 计算训练集的数据个数
    train_size = int((1-rate)*len(features_set))

    # 调用random_number获得随机数据索引
    data_index = random_number(len(features_set))

    # 分隔数据集
    # x是自变量，即输入（分类特征）；y是因变量，即输出（分类结果）
    x_train = features_set[data_index[:train_size]]
    x_test = features_set[data_index[train_size:]]

    y_train = labels_set[data_index[:train_size]]
    y_test = labels_set[data_index[train_size:]]

    return x_train, x_test, y_train, y_test


def data_distance(x_test, x_train):
    """
    :param x_test: 测试集
    :param x_train: 训练集
    :return: 返回计算的距离
    """
    distances = np.sqrt(sum((x_test - x_train) ** 2))
    return distances


def knn(x_train, y_train, x_test, k):
    """
    :param x_train: 训练集特征数据
    :param y_train: 训练集标签数据
    :param x_test: 测试集特征数据
    :param k: 邻居数
    :return: 返回一个列表包含预测结果
    """
    # 预测结果列表，用于存储测试集预测出来的结果
    predict_result_set = []

    # 训练集的长度
    train_set_size = len(x_train)

    # 创建一个全零的矩阵，长度为训练集的长度
    distances = np.array(np.zeros(train_set_size))

    # 计算每一个测试集与每一个训练集的距离
    for x in x_test:
        for index in range(train_set_size):
            # 计算数据之间的距离
            distances[index] = data_distance(x, x_train[index])

        # 排序后的距离的下标（从小到大）
        sorted_dist = np.argsort(distances)

        class_count = {}

        # 取出k个最短距离
        for j in range(k):
            # 获得下标所对应的标签值
            sort_label = y_train[sorted_dist[j]]

            # 将标签存入字典之中并存入个数
            class_count[sort_label] = class_count.get(sort_label, 0) + 1

        # 对标签进行排序
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

        # 将出现频次最高的放入预测结果列表
        predict_result_set.append(sorted_class_count[0][0])

    # 返回预测结果列表
    return predict_result_set


def predict_score(predict_result_set, y_test):
    """
    :param predict_result_set: 预测结果列表
    :param y_test: 测试集标签数据
    :return: 返回测试精度
    """
    count = 0
    for i in range(0, len(predict_result_set)):
        if predict_result_set[i] == y_test[i]:
            count = count + 1

    accuracy = count / len(predict_result_set)
    return accuracy


if __name__ == "__main__":
    # 1.读入数据
    iris_data_set = open_file("D:\\iris.csv")

    # 2.分割训练集和测试集
    x_train, x_test, y_train, y_test = split_data_set(iris_data_set[0], iris_data_set[1])

    # 3.调用KNN算法
    start_time = datetime.datetime.now()
    result = knn(x_train, y_train, x_test, 8)
    end_time = datetime.datetime.now()

    # 4.准确率
    acc = predict_score(result, y_test)

    print("正确标签：", y_test)
    print("预测结果：", np.array(result))
    print("准确率：" + str(acc * 100) + "%")
    # print("测试集的精度：%.2f" % acc)
    print("用时：" + str((end_time - start_time).microseconds / 1000) + 'ms')
