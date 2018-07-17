# encoding="utf-8"

import numpy as np

class PCA:
    """
    PCA 主成分分析
    算法：http://www.cnblogs.com/pinard/p/6239403.html
    实现：https://www.cnblogs.com/lzllovesyl/p/5235137.html
    测试矩阵：a = np.mat([[2.5,2.4], [0.5,0.7], [2.2,2.9], [1.9,2.2], [3.1,3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])
    Parameters:
    n_dimension: Dimension of low dimensional space after dimensionality reduction
    thre_value: the threshold that is used by selecting the minimum d' that satisfies it
    """
    def __init__(self, n_dimension=0, thre_value=0):
        self.n_dimension = n_dimension
        self.thre_value = thre_value

    def fit(self, x):
        #求协方差矩阵
        meanVals = np.mean(x, axis=0)   #求均值 axis 0:列均值 axis 1：行均值
        meanRemoved = x - meanVals  #数据中心化
        covMat = np.cov(meanRemoved, rowvar=False)  #协方差矩阵x.T * x rowvar：False 以列为单位   rowvar：True 以行为单位（默认）  

        eigVals, eigVectors = np.linalg.eig(np.mat(covMat)) #计算特征值，特征向量
        n_sorted_id = []
        sorted_id = np.argsort(eigVals) #特征值序号排序(升序) axis = 0 沿y轴方向排序 axis = 1 沿x轴方向排序
        if self.thre_value == 0:
            n_sorted_id =sorted_id[:-self.n_dimension-1:-1] #取前n个最大的特征值 序号
        else:
            sum_eigVals = np.sum(eigVals)
            for index in range(1, len(sorted_id) + 1):
                if np.sum(eigVals[sorted_id[:-index-1:-1]]) / sum_eigVals >= self.thre_value:
                    n_sorted_id = sorted_id[:-index-1:-1]
                    break
        #根据输入的维度，构建过渡矩阵wMat
        wMat = eigVectors[:, n_sorted_id[0]]
        for index in range(1, len(n_sorted_id)):
            wMat = np.c_[wMat, eigVectors[:, n_sorted_id[index]]]
        redMat = meanRemoved * wMat #降维映射矩阵
        return redMat

