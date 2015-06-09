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
acuracias=[]


#classificador knn

knn_acu_train=[]
knn_acu_test=[]
ks=[3,5,7,9,11]
for i in ks:
    teste=[]
    treino=[]
    for treino_ind, teste_ind in kf:
        knn_iris = KNeighborsClassifier(n_neighbors=i)
        knn_iris.fit(valores[treino_ind],classes[treino_ind])
        teste.append(accuracy_score(classes[teste_ind], knn_iris.predict(valores[teste_ind])))
        treino.append(accuracy_score(classes[treino_ind], knn_iris.predict(valores[treino_ind])))
    knn_acu_test.append(np.mean(teste))
    knn_acu_train.append(np.mean(treino))

plt.clf()
plt.plot(ks, knn_acu_test,c="red")
plt.plot(ks, knn_acu_train,c="blue")
plt.savefig('iris_knn.png')


acuracias.append(knn_acu_test[0])
k_usado=3
for i in range(1,len(knn_acu_test)):
    if knn_acu_test[i]>=knn_acu_train[i]:
        acuracias[0]=knn_acu_test[i]
        k_usado=k_usado+i*2

#classificador naive bayes
teste=[]
for treino_ind, teste_ind in kf:
    gnb_iris = GaussianNB()
    gnb_iris.fit(valores[treino_ind],classes[treino_ind])
    teste.append(accuracy_score(classes[teste_ind], gnb_iris.predict(valores[teste_ind])))
acuracias.append(np.mean(teste))

#classificador Decision Tree
teste=[]
for treino_ind, teste_ind in kf:
    clf_iris = tree.DecisionTreeClassifier()
    clf_iris.fit(valores[treino_ind],classes[treino_ind])
    teste.append(accuracy_score(classes[teste_ind], clf_iris.predict(valores[teste_ind])))
acuracias.append(np.mean(teste))

plt.clf()
plt.bar([1,2,3],acuracias)
plt.xlim(0.5, 4.5)
plt.ylim(0.8, 1)
plt.xticks([1.4,2.4,3.4],["Knn","Naive Bayes","Decision Tree"])
plt.savefig('iris_acuracias.png')

print "Fim"

