import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import MnistReadDemo as MR
# import ReadImageDemo as RI
from skimage import io, transform, data, color
import csv

timeS = time.time()

N = 28
class_num = 10
feature_len = 784
time_0 = time.time()
train_num = 60000
test_num = 10000
# ALLROWS = 60000

def all_np(arr):
    label_num = []
    arr = np.array(arr)
    key = np.unique(arr)
    # result = {}

    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        label_num.append(v)
    return label_num


def get_train_set():
    f = open('data.csv', 'wb')
    category = MR.read_label('train-labels.idx1-ubyte', 'train/label.txt')

    # file_names = os.listdir(r"./train/", )
    train_picture = np.zeros([train_num, N ** 2 + 1])
    # 遍历文件，转为向量存储
    for file in range(train_num):
        img_num = io.imread('./train/%d.png' % (file))
        # img_cut = RI.cutImage(img_num)
        # dst = transform.resize(img_num, (N, N), mode='constant', anti_aliasing=True)
        rows, cols = img_num.shape
        for i in range(rows):
            for j in range(cols):
                if img_num[i, j] < 100:
                    img_num[i, j] = 0
                else:
                    img_num[i, j] = 1
        train_picture[file, 0:N ** 2] = img_num.reshape(N ** 2)
        train_picture[file, N ** 2] = category[file]
        print("完成处理第%d张图片" % (file+1))
    np.savetxt(f,train_picture,fmt='%d',delimiter=',', newline='\n', header='', footer='')
    f.close()
    time_e = time.time()
    print('process data train cost ', time_e - time_0, ' seconds', '\n')
    return train_picture

def get_test_set():
    f = open('dataTest.csv', 'wb')
    time_t = time.time()
    # 读取num目录下的所有文件名
    # fileNames = os.listdir(r"./test/", )

    category_test = MR.read_label('t10k-labels.idx1-ubyte', 'test/label.txt')
    test_picture = np.zeros([test_num, N ** 2 + 1])
    # 遍历文件，转为向量存储
    for file in range(test_num):
        img_num = io.imread('./test/%d.png' % (file))
        # img_cut = RI.cutImage(img_num)
        # dst = transform.resize(img_cut, (N, N), mode='constant', anti_aliasing=True)
        rows, cols = img_num.shape
        for i in range(rows):
            for j in range(cols):
                if img_num[i, j] < 100:
                    img_num[i, j] = 0
                else:
                    img_num[i, j] = 1
        test_picture[file, 0:N ** 2] = img_num.reshape(N ** 2)
        test_picture[file, N ** 2] = category_test[file]
        print("完成处理第%d张图片" % (file + 1))
    np.savetxt(f, test_picture, fmt='%d', delimiter=',', newline='\n', header='', footer='')
    f.close()
    time_e = time.time()
    print('process data test cost ', time_e - time_t, ' seconds', '\n')
    return test_picture


def loadCSVfile(filename):
    tmp = np.loadtxt(filename, delimiter=",")
    dataTmp = tmp[:, 0:N**2].astype(np.int)  # 加载数据部分
    labelTmp = tmp[:, N**2].astype(np.int)  # 加载类别标签部分
    print(tmp.shape)
    return dataTmp, labelTmp  # 返回array类型的数据



# def getXKProbability():
#     element_count_zero = []
#     element_count_one = []
#     for col in range(cols):
#         colArray = data_map[:, col]
#         # arr = np.array(colArray)
#         mask_one = (colArray == 1)
#         arr_new_one = colArray[mask_one]
#         v1 = arr_new_one.size + 1
#         element_count_one.append(v1)
#
#         mask_zero = (colArray == 0)
#         arr_new_zero = colArray[mask_zero]
#         v0 = arr_new_zero.size + 1
#         element_count_zero.append(v0)
#     return element_count_one, element_count_zero
#
# element_count_one, element_count_zero = getXKProbability()
#
# # element_probability = np.array(element_count) / ALLROWS
#
# element_probability = np.concatenate([np.array(element_count_zero) / (ALLROWS+2), np.array(element_count_one) / (ALLROWS+2)], axis=0).reshape((2,-1))
#
# print(np.sum(element_probability), element_probability)


# p(xk=i|y=j)表示样本属于第j类情况下第k个元素的值为i的概率
# label = np.array(category)
# key = np.unique(label)

# element_XK_probability = np.zeros([10, N**2, 2])
#
# for k in key:
#     label_K_index = np.where(label == k)[0]
#     # print(k, len(label_K_index), label_K_index)
#     for col in range(cols):
#         colArray = data_map[:, col]
#         colTemp = colArray[label_K_index]
#         mask = (colTemp == 1)
#         arr_new = colTemp[mask]
#         # print(k, col, mask)
#         v = arr_new.size + 1
#         element_XK_probability[k, col, 0] = v / (len(label_K_index)+1)*1000
#         element_XK_probability[k, col, 1] = (1-(v / (len(label_K_index)+1)))*1000
# print(element_XK_probability)

def Train():
    conditional_probability = np.zeros((class_num, feature_len, 2))   # 条件概率

    # 计算先验概率及条件概率
    for i in range(len(labels)):
        img = data_map[i, :]
        label = labels[i]
        for j in range(feature_len):
            conditional_probability[label][j][img[j]] += 1

    # 将概率归到[1.10001]
    for i in range(class_num):
        for j in range(feature_len):

            # 经过二值化后图像只有0，1两种取值
            pix_0 = conditional_probability[i][j][0]
            pix_1 = conditional_probability[i][j][1]

            # 计算0，1像素点对应的条件概率
            probalility_0 = (float(pix_0)/float(pix_0+pix_1))*1000000 + 1
            probalility_1 = (float(pix_1)/float(pix_0+pix_1))*1000000 + 1

            conditional_probability[i][j][0] = probalility_0
            conditional_probability[i][j][1] = probalility_1

    return conditional_probability


# 计算概率
def calculate_probability(img, label):
    probability = int(prior_probability[label])

    for i in range(len(img)):
        probability *= int(conditional_probability[label][i][img[i]])

    return probability

def Predict(testset, test_labels):
    predict = []
    accuracy = []
    right = 0
    rows, cols = testset.shape
    for row in range(rows):
        # 图像二值化
        img = testset[row, :]

        max_label = 0
        max_probability = calculate_probability(img, 0)

        for j in range(1, 10):
            probability = calculate_probability(img, j)

            if max_probability < probability:
                max_label = j
                max_probability = probability
        predict.append(max_label)
        if max_label == test_labels[row]:
            right += 1
        if (row+1) % 500 == 0:
            accuracy.append(float(right)/(row+1))
    return float(right)/len(test_labels), np.array(predict), accuracy


if __name__ == '__main__':
    print('Start process train data')
    time_0 = time.time()
    # get_train_set()

    print('Start process test data')
    time_t = time.time()
    get_test_set()

    print ('Start read train data')
    time_1 = time.time()
    data_map, labels = loadCSVfile("data.csv")
    print(data_map.shape, labels.shape)
    time_2 = time.time()
    print('read data train cost ', time_2 - time_1, ' seconds', '\n')

    print('Start training')
    prior_probability = all_np(labels)
    conditional_probability = Train()
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' seconds', '\n')

    print('Start read predict data')
    time_4 = time.time()
    test_data_map, test_labels = loadCSVfile("dataTest.csv")
    print(test_data_map.shape, test_data_map.shape)
    time_5 = time.time()
    print('read predict data cost ', time_5 - time_4, ' seconds', '\n')
    print('Start predicting')
    score, test_predict, accuracy = Predict(test_data_map, test_labels)
    time_6 = time.time()
    print('predicting cost ', time_6 - time_5, ' seconds', '\n')

    new_ticks = np.linspace(1, 20, 20)
    plt.xticks(new_ticks)
    plt.plot(new_ticks, accuracy, 'o-', color='g')
    plt.xlabel("x -- 1:500")
    plt.ylabel("y")
    plt.title(u"预测准确率")
    plt.show()

    print("The accuracy rate is ", score)
    print("All data processing cost %s seconds" % (time_6 - time_0))
