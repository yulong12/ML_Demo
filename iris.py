from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()
#得到分类器
iris = datasets.load_iris()
#加载数据集

# save data
# f = open("iris.data.csv", 'wb')
# f.write(str(iris))
# f.close()

print (iris)

knn.fit(iris.data, iris.target)
#训练模型

predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
#预测新的实例

print ("hello")
# #print ("predictedLabel is :" + predictedLabel)
print (predictedLabel)