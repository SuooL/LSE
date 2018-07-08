import os
from skimage import io
import numpy as np
import cv2

##Essential vavriable 基础变量
#Standard size 标准大小
N = 100
#Gray threshold 灰度阈值
color = 120/255

#
# img = io.imread('./train/0.png')
# print(type(img))
# print(img.shape)  # 图片通道数
#
# cv_img = img.astype(np.uint8)
# cv2.threshold(cv_img, 50, 1, cv2.cv.CV_THRESH_BINARY_INV, cv_img)

# rows, cols = img.shape
# for i in range(rows):
#     for j in range(cols):
#         if img[i, j] < 128:
#             img[i, j] = 0
#         else:
#             img[i, j] = 255

# io.imshow(img_num)
# io.show()



def JudgeEdge(img, length, flag, size):
    '''Judge the Edge of Picture判断图片切割的边界'''
    for i in range(length):
        #Row or Column 判断是行是列
        if flag == 0:
            # Positive sequence 正序判断该行是否有手写数字
            line1 = img[i, img[i, :] > 0]
            # Negative sequence 倒序判断该行是否有手写数字
            line2 = img[length-1-i, img[length-1-i,:]>0]
        else:
            line1 = img[img[:,i]>0, i]
            line2 = img[img[:,length-1-i]>0,length-1-i]
        # If edge, recode serial number 若有手写数字，即到达边界，记录下行
        if len(line1)>=1 and size[0]==-1:
            size[0] = i
            # print("前端到达：",i)
        if len(line2)>=1 and size[1]==-1:
            size[1] = length-1-i
            # print("尾端到达：",i)
        # If get the both of edge, break 若上下边界都得到，则跳出
        if size[0]!=-1 and size[1]!=-1:
            break
    return size

def cutImage(img):
    '''Cut the Picture 切割图象'''
    # 初始化新大小
    size = []
    # 图片的行数
    length = len(img)
    #图片的列数
    width = len(img[0,:])
    # 计算新大小
    size.append(JudgeEdge(img, length, 0, [-1, -1]))
    size.append(JudgeEdge(img, width, 1, [-1, -1]))
    size = np.array(size).reshape(4)
    return img[size[0]:size[1]+1, size[2]:size[3]+1]