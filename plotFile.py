# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.rcParams['font.family']=['PingFang']

x_cord = []
y_cord = []
def drawScatterDiagram(fileName):
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.split(',')
        x_cord.append(float(lineArr[0]))
        y_cord.append(float(lineArr[1]))
    # plt.scatter(x_cord,y_cord,s=30,c='red',marker='o', alpha=0.7,label='比赛成绩 ')
    # plt.xlabel("year")
    # plt.ylabel("time")
    # plt.title("result of game")

def linearCalculate():
    x = np.array(x_cord)
    y = np.array(y_cord)
    x_mean = np.mean(x_cord)
    y_mean = np.mean(y_cord)
    xy_mean = np.mean(x*y)
    x_square_mean = np.mean(x**2)

    w1 = (xy_mean-x_mean*y_mean)/(x_square_mean-x_mean**2)
    w0 = y_mean - w1*x_mean
    xasix = np.linspace(1896, 2008, 112)
    yasix = w1 * xasix + w0
    # print(w0, w1)
    plt.plot(xasix,yasix, label='线性拟合-Scalar')
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

def noLinearCalculate():
    x_zeros = np.ones((len(x_cord),1))
    x_mat = np.mat(np.c_[x_zeros,x_cord])
    y_mat = np.mat(y_cord).T
    w_mat = ((x_mat.T * x_mat).I*x_mat.T*y_mat)[::-1]
    x_mLo = np.linspace(1896, 2008, 112)
    # y_mLo = w_mat[1,0] * x_mLo + w_mat[0,0]
    c = np.squeeze([i for i in w_mat])
    print(c.T)
    func = np.poly1d(c.T)
    y_mLo = func(x_mLo)
    plt.plot(x_mLo,y_mLo, c="green",label='线性拟合-Matrix')
    plt.legend(loc='upper right')

def noLinearMoreTimesCalculate():
    x_mat = np.ones((len(x_cord),1))
    y_mat = np.mat(y_cord).T
    test = np.array(x_cord)
    test = (test - 1896)/4.0
    # x_mat = np.mat(np.c_[x_zeros,test])
    for index in range(1,50):
        x_temp = test**index
        x_mat = np.mat(np.c_[x_mat, x_temp])
    print(x_mat)
    w_mat = ((x_mat.T * x_mat).I*x_mat.T*y_mat)[::-1]
    w_mat = w_mat.T
    c = np.squeeze([i for i in w_mat])
    print(c)
    func = np.poly1d(c)
    x_mLo = np.linspace(0, 27, 112)

    y_mLo = func(x_mLo)

    plt.scatter(test,y_cord,s=30,c='red',marker='o', alpha=0.7,label='比赛成绩 ')
    plt.xlabel("year")
    plt.ylabel("time")
    plt.title("result of game")

    plt.plot(x_mLo,y_mLo, c="yellow",label='线性拟合-8阶')
    plt.legend(loc='upper right')

if __name__ == '__main__':
    drawScatterDiagram("olympic100m.txt")
    # linearCalculate()
    # noLinearCalculate()
    noLinearMoreTimesCalculate()
    plt.show()