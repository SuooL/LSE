# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(-5000,5000, size=[100,1]) / 1000.0

t = 5*(x**3)  - x**2 + x + 100*np.random.randn(100, 1);

testx = np.linspace(-5,5,1001)

testt = 5*(testx**3) - testx**2 + testx + 100*np.random.randn(1001, 1)
plt.figure()
plt.subplot(2,1,1)
plt.scatter(x, t, s=35, c="red", marker='o', alpha=0.9)
plt.xlabel("x")
plt.ylabel("y")
plt.title("points of train")

x_mat = np.ones((len(x),1))
y_mat = np.mat(t)

color = ["","",""]

degrees = np.arange(2, 10)
trainLoss = []
predictLoss = []
LOOCVLoss = []

def trainAndGetLoss():
    for degree in degrees:
        global x_mat
        global y_mat
        tempX = x_mat
        tempY = y_mat
        for index in range(1, degree):
            x_temp = x ** index
            tempX = np.mat(np.c_[tempX, x_temp])
        # print(tempX)
        print("计算")
        print(type(tempX.T*tempX))
        w_mat = ((tempX.T * tempX).I * tempX.T * tempY)[::-1]
        w_mat = w_mat.T
        c = np.squeeze([i for i in w_mat])
        print(c)
        func = np.poly1d(c)
        if degree ==2 or degree == 3 or degree == 5 :
            x_mLo = np.linspace(-5, 5, 100)
            y_mLo = func(x_mLo)
            plt.subplot(2, 1, 1)
            plt.plot(x_mLo, y_mLo, linewidth=2, c=np.random.rand(3,), label='线性拟合-%d阶' % (degree - 1))
            plt.legend(loc='upper left')
        #  训练数据损失
        y_train = func(x)
        print(np.mean((y_train-t)**2), type(y_train), len(y_train))
        trainLoss.append(np.mean((y_train-t)**2))
        # 预测数据损失
        y_predict = func(testx)
        print(np.mean((y_predict-testt)**2), type(y_predict), len(y_predict))
        predictLoss.append(np.mean((y_predict-testt)**2))
    plt.subplot(2,3,4)
    new_ticks = np.linspace(1, 8, 8)
    plt.xticks(new_ticks)
    plt.plot(new_ticks, trainLoss, 'o-', color='g')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(u"训练数据损失")

    plt.subplot(2,3,5)
    plt.xticks(new_ticks)
    plt.plot(new_ticks, predictLoss, 'o-', color='g')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(u"预测数据损失")

def trainLOOCVAndGetLoss():
    for degree in degrees:
        trainLoocv = []
        for index in range(0, 100):
            # 准备训练数据
            index_x = (index)
            new_x = np.delete(x, index_x)
            new_y = np.delete(t, index_x)
            tempX = np.ones([99, 1])
            tempY = np.mat(new_y).T
            for index in range(1, degree):
                x_temp = new_x ** index
                tempX = np.mat(np.c_[tempX, x_temp])
            w_mat = ((tempX.T * tempX).I * tempX.T * tempY)[::-1]
            w_mat = w_mat.T
            c = np.squeeze([i for i in w_mat])
            func = np.poly1d(c)
            if index_x == 0:
                print((func(x[index,0])-t[index,0])**2)
            trainLoocv.append((func(x[index,0])-t[index,0])**2)
        LOOCVLoss.append(np.mean(trainLoocv))
        plt.subplot(2, 3, 6)
        new_ticks = np.linspace(1, 8, 8)
        plt.xticks(new_ticks)
        plt.plot(new_ticks, predictLoss, 'o-', color='g')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("LOOCV损失")


if __name__ == '__main__':
    trainAndGetLoss()
    trainLOOCVAndGetLoss()
    plt.show()

