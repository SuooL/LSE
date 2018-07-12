# -*- coding: utf-8 -*
import time
import matplotlib.pyplot as plt
import testLibrary as tl
import collections
import numpy as np

# 距离计算
def calc_dis(train_image,test_image):
    dist=np.linalg.norm(train_image-test_image)
    return dist


# 确定待分类实例的 k 近邻
def find_labels(k,train_images,train_labels,test_image):
    all_dis = []
    labels=collections.defaultdict(int)
    for i in range(len(train_images)):
        dis = np.linalg.norm(train_images[i]-test_image)
        all_dis.append(dis)
    sorted_dis = np.argsort(all_dis)
    count = 0
    while count < k:
        labels[train_labels[sorted_dis[count]]]+=1
        count += 1
    return labels


# 结合训练数据集，对所有待分类实例进行 k 近邻分类预测
def knn_all(k,train_images,train_labels,test_images, test_labels):
    print("start knn_all!")
    res=[]
    right = 0
    accuracy = []
    count=0
    for i in range(2100):
        labels=find_labels(k,train_images,train_labels,test_images[i])
        res.append(max(labels))
        print("Picture %d has been predicted! real is %d predicted is %d"%(count, test_labels[i], max(labels)))
        count+=1
        if max(labels) == test_labels[i]:
            right+=1
        if (i+1) % 70 == 0:
            accuracy.append(float(right)/(i+1))
    return res, accuracy

# 总的预测准确率计算
def calc_precision(res,test_labels):
    f_res_open=open("res.txt","a+")
    precision=0
    for i in range(len(res)):
        f_res_open.write("res:"+str(res[i])+"\n")
        f_res_open.write("test:"+str(test_labels[i])+"\n")
        if res[i]==test_labels[i]:
            precision+=1
    return precision/len(res)


if __name__ == '__main__':
    print('Start process train data')
    time_0 = time.time()
    # tl.get_train_set()

    print('Start process test data')
    time_t = time.time()
    # tl.get_test_set()

    # 读取训练数据集和测试数据集的方法和朴素贝叶斯方法一致
    print ('Start read train data')
    time_1 = time.time()
    data_map, labels = tl.loadCSVfile("data.csv")
    print(data_map.shape, labels.shape)
    time_2 = time.time()
    print('read data train cost ', time_2 - time_1, ' seconds', '\n')

    print('Start read predict data')
    time_3 = time.time()
    test_data_map, test_labels = tl.loadCSVfile("dataTest.csv")
    print(test_data_map.shape, test_data_map.shape)
    time_4 = time.time()
    print('read predict data cost ', time_4 - time_3, ' seconds', '\n')

    print('Start predicting data')
    time_5 = time.time()
    res, accuracy = knn_all(3, data_map, labels, test_data_map, test_labels)
    score = calc_precision(res, test_labels)
    time_6 = time.time()
    print('read predict data cost ', time_6 - time_5, ' seconds', '\n')

    new_ticks = np.linspace(1, 30, 30)
    plt.xticks(new_ticks)
    plt.ylim(ymin=0.5, ymax = 1)
    plt.plot(new_ticks, accuracy, 'o-', color='g')
    plt.xlabel("x -- 1:70")
    plt.ylabel("y")
    plt.title(u"预测准确率")
    plt.show()

    print("The accuracy rate is ", score)
    print("All data processing cost %s seconds" % (time_6 - time_0))