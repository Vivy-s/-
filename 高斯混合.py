
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats
import em


np.random.seed(100)  # 固定随机数种子，确保下次运行数据相同

#产生满足正太分布的随机数，参数分别为：均值，方差，样本量
male=np.random.normal(180,np.sqrt(10),100)
female=np.random.normal(163,np.sqrt(10),100)

h=list(male)# 转化为list
h.extend(female)
h=np.array(h)
#GMM的构造
#Step 1    均值、方差和权值进行初始化
mu1=170  #均值
sigmal=10  #方差
w1=0.7   #权值

mu2=160
sigma2=10
w2=0.3

d=1
n = len(h)  # 样本长度

#EM算法
def em_new(h, mu1, sigmal, w1, mu2, sigma2, w2):
    d = 1
    n = len(h)  # 样本长度

    # E-step
    # 计算响应
    # p1=w1*flot(stats.norm(mu1,sigmal))
    p1 = w1 * stats.norm(mu1, sigmal).pdf(h)
    p2 = w2 * stats.norm(mu2, sigma2).pdf(h)

    # p1, p2权重 * 后验概率
    R1i = p1 / (p1 + p2)
    R2i = p2 / (p1 + p2)

    # M-step
    # mu的更新
    mu1 = np.sum(R1i * h) / np.sum(R1i)
    mu2 = np.sum(R2i * h) / np.sum(R2i)
    # sigmal的更新
    sigmal = np.sqrt(np.sum(R1i * np.square(h - mu1)) / (d * np.sum(R1i)))
    sigma2 = np.sqrt(np.sum(R2i * np.square(h - mu2)) / (d * np.sum(R2i)))
    # w的更新
    w1 = np.sum(R1i) / n
    w2 = np.sum(R2i) / n

    return mu1, sigmal, w1, mu2, sigma2, w2

for iteration in range(10):
    mu1,sigmal,w1,mu2,sigma2,w2=em_new(h,mu1,sigmal,w1,mu2,sigma2,w2)
    print('均值1：')
    print(mu1)

    print('方差1：')
    print(sigmal)

    print('权重1：')
    print(w1)

    print('均值2：')
    print(mu2)

    print('方差2：')
    print(sigma2)

    print('权重2：')
    print(w2)
