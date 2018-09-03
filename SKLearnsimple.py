from sklearn import svm

#x是（2，0），（1，1），（2，3）三个点
x = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]
clf = svm.SVC(kernel = 'linear')

clf.fit(x, y)


print("==========clf===========")
print (clf)
print("=======clf.support_vectors_==============")
# get support vectors
print (clf.support_vectors_)
# get indices of support vectors
print("========clf.support_=============")
print (clf.support_)
# get number of support vectors for each class
print("=======clf.n_support_==============")
print (clf.n_support_)

print("==========predict===========")
#预测点（2，0）

print(clf.predict([[2,0]]))