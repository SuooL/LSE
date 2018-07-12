# coding=utf-8
from numpy import *


class Data(object):
    def __init__(self, attributeList):
        """
        :type attributeList: list
        """
        self.attributeList = []
        for attribute in attributeList:
            if attribute is not attributeList[-1]:
                attribute = float(attribute)
            self.attributeList.append(attribute)


class Node(object):
    def __init__(self, axis, depth):
        self.father = None #指向父节点
        self.leftSub = None #指向左子节点
        self.rightSub = None #指向右子节点
        self.axis = axis #与超平面垂直的轴
        self.dataInNode = set()# 用于存放保存在该节点的样本
        self.depth = depth #节点的深度
        self.label = None #节点中样本最多的类别（不仅仅是保存在该节点的样本，还包括保存在该节点的子节点的样本）
        self.mid = 0 #切分节点时的轴上坐标
        self.flag = 0 #标记该节点有没有被搜索过


class TestData(object):
    def __init__(self, attributeList):
        """
        :type attributeList: list
        """
        self.distance = None
        self.attributeList = []
        for attribute in attributeList:
            if attribute is not attributeList[-1]:
                attribute = float(attribute)
            self.attributeList.append(attribute)
        self.label = None

    def distanceCount(self, data):
        dataAttributeMat = mat(data.attributeList[:-1])
        targetMat = mat(self.attributeList[:-1])
        # 用样本点的属性值生成特征向量
        distance = sqrt(float(sum((dataAttributeMat - targetMat).transpose() * (dataAttributeMat - targetMat))))
        # 计算该测试样本点与训练样本点data的欧氏距离
        return distance

    def distanceToHyperPlane(self, axis, mid):
        targetMat = mat(self.attributeList[:-1])
        m, n = shape(targetMat)
        planeVector = zeros(n)
        planeVector = mat(planeVector)
        planeVector[0, axis] = 1
        distance = abs(planeVector * targetMat.transpose() - mid)
        return distance[0, 0]


def loadDataSet():
    fr = open(r'data\mlia\Ch02\datingTestSet.txt')
    dataSet = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        data = Data(lineArr)
        dataSet.append(data)
    return dataSet


def midCount(midCountList):
    midCountList.sort()
    length = len(midCountList)
    if length % 2 == 0:
        mid = float(midCountList[length / 2 - 1] + midCountList[length / 2]) / 2
    else:
        mid = midCountList[(length - 1) / 2]
    return mid


def treeGenerate(dataSet, currentNode, fatherNode):
    currentNode.father = fatherNode
    # 将当前节点与父节点连接
    if dataSet is None:
        # 如果传入样本集为空
        return None
    length = len(dataSet)
    # 获得样本集中的样本数目
    labelCountDict = {}
    for data in dataSet:
        if data.attributeList[-1] not in list(labelCountDict.keys()):
            labelCountDict[data.attributeList[-1]] = 1
        else:
            labelCountDict[data.attributeList[-1]] += 1
    # 对样本集中的样本类别计数
    if len(labelCountDict) == 1:
        # 如果样本集中所有样本都是同一类别，则将样本保存到该节点，将该节点设为叶节点并返还
        for data in dataSet:
            currentNode.dataInNode.add(data)
        currentNode.label = max(labelCountDict)
        # 设定叶节点的类别
        return currentNode
    if length >= 2:
        # 如果有不少于两个不同类别的样本在样本集中
        axis = currentNode.axis
        # 获得切分的维度
        midCountList = []
        for data in dataSet:
            midCountList.append(data.attributeList[axis])
        mid = midCount(midCountList)
        currentNode.mid = mid
        # 计算该维度上的特征值的中位数,并将其保存到节点
        leftDataSet = []
        rightDataSet = []
        labelCountDict = {}
        for data in dataSet:
            if data.attributeList[axis] == mid:
                currentNode.dataInNode.add(data)
                if data.attributeList[-1] not in labelCountDict:
                    labelCountDict[data.attributeList[-1]] = 1
                else:
                    labelCountDict[data.attributeList[-1]] += 1
            elif data.attributeList[axis] < mid:
                leftDataSet.append(data)
            else:
                rightDataSet.append(data)
        # 切分样本集，落在超平面上的样本保存在本节点，小于的划分到左子节点，大于的划分到右子节点
        if len(labelCountDict) != 0:
            currentNode.label = max(labelCountDict)
        # 设定当前节点的类别为样本集中最多的类别
        axis = currentNode.depth % (len(dataSet[0].attributeList) - 1)
        # 结算下一个切分维度 下一个切分维度=当前节点深度（mod 样本属性总数）+1
        leftNode = Node(axis, currentNode.depth + 1)
        rightNode = Node(axis, currentNode.depth + 1)
        currentNode.leftSub = treeGenerate(leftDataSet, leftNode, currentNode)
        currentNode.rightSub = treeGenerate(rightDataSet, rightNode, currentNode)
        # 递归生成左右子节点
        return currentNode


def createTree():
    dataSet = loadDataSet()
    root = Node(0, 1)
    root = treeGenerate(dataSet, root, None)
    return root


def loadTarget():
    fr = open(r'data\mlia\Ch02\test.txt')
    targetSet = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        target = TestData(lineArr)
        if len(target.attributeList) != 0:
            targetSet.append(target)
    return targetSet


def searchLeaf(target, node):
    """
    :type node: Node
    :type target: TestData
    """
    if target is None:
        return
    if node.leftSub is None and node.rightSub is None:
        target.label = node.label
        return node
    # 如果当前节点左右子节点为空，则返还当前节点
    axis = node.axis
    # 获取当前节点的切分轴
    mid = node.mid
    # 获得当前节点的切分中位数
    if target.attributeList[axis] < mid:
        leafNode = searchLeaf(target, node.leftSub)
        return leafNode
    else:
        leafNode = searchLeaf(target, node.rightSub)
        return leafNode


def searchLeaves(targetSet, root):
    """
    :type root: Node
    :type targetSet: List
    """
    leafNodes = {}
    for target in targetSet:
        leafNodes[target] = searchLeaf(target, root)
    return leafNodes


def searchData(target, node):
    """
    :type node: Node
    :type target: TestData
    """
    if target is None or node is None:
        return
    node.flag = 1
    # 标记该节点已被搜寻过
    distanceCount = {}
    for data in node.dataInNode:
        distance = target.distanceCount(data)
        distanceCount[data] = distance
        # 计算测试样本点与节点中每个样本点的距离
    if target.distance is None:
        data = min(distanceCount)
        target.label = data.attributeList[-1]
        target.distance = target.distanceCount(data)
        # 如果测试样本点尚未匹配当前最近点，则匹配最近样本点，并将该样本点的距离和类别存入测试样本点
    else:
        if len(distanceCount) != 0:
            if target.distance > min(distanceCount):
                data = min(distanceCount)
                target.label = data.attributeList[-1]
                target.distance = target.distanceCount(data)
                # 如果找到比当前最近点更近的样本点，则将该样本点设为当前最近点
    father = node.father
    # 进入父节点
    if father is None:
        # 如果搜索到了根节点
        return target
    axis = father.axis
    mid = father.mid
    distance = target.distanceToHyperPlane(axis, mid)
    # 计算当前样本点到父节点超平面距离
    if distance > target.distance or (father.leftSub.flag != 0 and father.rightSub.flag != 0):
        # 如果样本点到当前最近点的距离比到超平面的距离小,或者该父节点的所有子节点已经被搜索过
        target = searchData(target, father)
        # 搜索父节点
        return target
    else:
        # 如果样本点到当前最近点的距离比到超平面的距离大
        if father.leftSub.flag == 0:
            # 锁定父节点的另一个非当前节点的子节点
            leafNode = searchLeaf(target, father.leftSub)
            # 向下搜索该节点直到叶节点
            target = searchData(target, leafNode)
            # 从搜索到的叶节点向上搜索
            return target
        elif father.rightSub.flag == 0:
            leafNode = searchLeaf(target, father.rightSub)
            # 向下搜索该节点直到叶节点
            target = searchData(target, leafNode)
            # 从搜索到的叶节点向上搜索
            return target


def test():
    root = createTree()
    targetSet = loadTarget()
    leafNodes = searchLeaves(targetSet, root)
    resultCount = {'right': 0, 'wrong': 0}
    for target, node in list(leafNodes.items()):
        target = searchData(target, node)
        if target.label == target.attributeList[-1]:
            resultCount['right'] += 1
        else:
            resultCount['wrong'] += 1
    print('正确率为:', str(float(resultCount['right']) / (resultCount['right'] + resultCount['wrong'])*100)+"%")

test()