import numpy as np
np.random.seed(1)

#Classificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

from sklearn.metrics import accuracy_score # Para calcualr a acuracia
from sklearn import cross_validation       # Para usar o kfold

import matplotlib.pyplot as plt

#Abrindo dados com numpy
valores = np.loadtxt("iris.data", delimiter=',', usecols=[0,1,2,3])
classes = np.loadtxt("iris.data", delimiter=',',dtype='|S15', usecols=[4])

#usando k-fold
kf = cross_validation.KFold(len(classes), n_folds=10)
saida=[]

#classificador knn
for treino_ind, teste_ind in kf:
    knn_iris = KNeighborsClassifier()
    knn_iris.fit(valores[treino_ind],classes[treino_ind])
    saida.append(accuracy_score(classes[teste_ind], knn_iris.predict(valores[teste_ind])))

plt.scatter(np.random.normal(1, 0.025, 10) , saida)

saida=[]
#classificador naive bayes
for treino_ind, teste_ind in kf:
    gnb_iris = GaussianNB()
    gnb_iris.fit(valores[treino_ind],classes[treino_ind])
    saida.append(accuracy_score(classes[teste_ind], gnb_iris.predict(valores[teste_ind])))

plt.scatter(np.random.normal(2, 0.025, 10),saida)


#classificador Decision Tree
saida=[]
for treino_ind, teste_ind in kf:
    clf_iris = tree.DecisionTreeClassifier()
    clf_iris.fit(valores[treino_ind],classes[treino_ind])
    saida.append(accuracy_score(classes[teste_ind], clf_iris.predict(valores[teste_ind])))

plt.scatter(np.random.normal(3, 0.025, 10),saida)

plt.xlim(0, 4)
plt.ylim(0, 1.1)
plt.xticks([1,2,3],["Knn","Naive Bayes","Decision Tree"])
plt.show()


teste = np.loadtxt("iris.data", delimiter=',',dtype={'names': ('sepal length', 'sepal width', 'petal length', 'petal width', 'label'),'formats': (np.float, np.float, np.float, np.float, '|S15')})

print teste['label']
print teste[['sepal length', 'sepal width', 'petal length', 'petal width']]
