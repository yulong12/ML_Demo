import numpy as np
import random

#numpoints：多少个实例，即多少行
#varivance：方差

def genData(numPoints,bias,variance):
    x = np.zeros(shape=(numPoints,2))#产生一个行数是numPoints列数是2的元素全是0的矩阵
    y = np.zeros(shape=(numPoints))#只有一列
    for i in range(0,numPoints):#包含0但是不包含numPoints
        x[i][0]=1#每一行的第一列都是1
        x[i][1]=i#每一行的第二列是i
        y[i]=(i+bias)+random.uniform(0,1)*variance
    return x,y

#alpha学习率，m:总共有m个实例，numIterations迭代的次数
def gradientDescent(x,y,theta,alpha,m,numIterations):
    xTran = np.transpose(x)#转置
    for i in range(numIterations):
        hypothesis = np.dot(x,theta)#内积
        loss = hypothesis-y#预测出来的值减去实际值
        cost = np.sum(loss**2)/(2*m)
        gradient=np.dot(xTran,loss)/m
        theta = theta-alpha*gradient
        print ("Iteration %d | cost :%f" %(i,cost))
    return theta

x,y = genData(100, 25, 10)
print("x:")
print(x)
print("y:")
print(y)

m,n = np.shape(x)#返回的是行数和列数
n_y = np.shape(y)

print("m:"+str(m)+" n:"+str(n)+" n_y:"+str(n_y))

numIterations = 100000
alpha = 0.0005
theta = np.ones(n)
theta= gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)

#输出的值是theta0,theta1