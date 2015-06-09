import numpy as np
np.random.seed(1)
import pandas as pd
import matplotlib.pyplot as plt

#Classificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score # Para calcualr a acuracia
from sklearn import cross_validation       # Para usar o kfold

##################################
## Funcoes
##################################
def arvore_de_decisao(classes,valores,kf):
    teste=[]
    for treino_ind, teste_ind in kf:
        treino_ind=list(treino_ind)
        teste_ind=list(teste_ind)
        clf = tree.DecisionTreeClassifier()
        clf.fit(valores.iloc[treino_ind,:],classes.iloc[treino_ind])
        teste.append(accuracy_score(classes.iloc[teste_ind], clf.predict(valores.iloc[teste_ind,:])))
    return np.mean(teste)

def naive_bayes(classes,valores,kf):
    teste=[]
    for treino_ind, teste_ind in kf:
        treino_ind=list(treino_ind)
        teste_ind=list(teste_ind)
        gnb = GaussianNB()
        gnb.fit(valores.iloc[treino_ind,:],classes.iloc[treino_ind])
        teste.append(accuracy_score(classes.iloc[teste_ind], gnb.predict(valores.iloc[teste_ind,:])))
    return np.mean(teste)

def knn(classes,valores,kf):
    


##################################
## Iris
##################################
#Abrindo dados com numpy
iris=pd.read_csv('iris/iris.data',header=None)

valores = iris.iloc[:,0:4]
classes = iris.iloc[:,4]

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
    treino_ind=list(treino_ind)
    teste_ind=list(teste_ind)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(valores.iloc[treino_ind,:],classes.iloc[treino_ind])
    teste.append(accuracy_score(classes.iloc[teste_ind], knn.predict(valores.iloc[teste_ind,:])))
    treino.append(accuracy_score(classes.iloc[treino_ind], knn.predict(valores.iloc[treino_ind,:])))
    knn_acu_test.append(np.mean(teste))
    knn_acu_train.append(np.mean(treino))
  return [knn_acu_test,knn_acu_train,ks]

plt.clf()
plt.scatter(ks, knn_acu_test,c="red")
plt.plot(ks, knn_acu_test,c="red")
plt.scatter(ks, knn_acu_train,c="blue")
plt.plot(ks, knn_acu_train,c="blue")
plt.savefig('knn.png')

acuracias.append(max(knn_acu_test))
k_usado=ks[knn_acu_test.index(max(knn_acu_test))]


#classificador naive bayes
acuracias.append(naive_bayes(classes,valores,kf))

#classificador Decision Tree
acuracias.append(arvore_de_decisao(classes,valores,kf))

plt.clf()
plt.bar([1,2,3],acuracias)
plt.xlim(0.5, 4.5)
plt.ylim(0.8, 1)
plt.xticks([1.4,2.4,3.4],["Knn","Naive Bayes","Decision Tree"])
plt.savefig('iris/acuracias.png')

print "Fim"

#Abrindo dados com numpy

car=pd.read_csv('car_evaluation/car.data',header=None)

print car.iloc[0:5,:]

valores = car.iloc[:,0:6]
classes = car.iloc[:,6]

#usando k-fold
kf = cross_validation.KFold(len(classes), n_folds=10)
acuracias=[]

#classificador knn
knn_acu_test,knn_acu_train,ks=knn(classes,valores,kf)
