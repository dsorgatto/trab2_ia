import numpy as np
np.random.seed(1)
import pandas as pd
import matplotlib.pyplot as plt

# Classificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score  # Para calcualr a acuracia
from sklearn import cross_validation  # Para usar o kfold

##################################
# # Iris
##################################
# Abrindo dados com panda
diabetes = pd.read_csv('pima-indians-diabetes.data', header=None)

valores = diabetes.iloc[:, 0:8]
#normalizando os valores
valores = (valores - valores.mean()) / (valores.std())
classes = diabetes.iloc[:, 8]

# usando k-fold
kf = cross_validation.KFold(len(classes), n_folds=10)
acuracias_media = []
acuracias_std = []
tabela = [0]

# classificador knn
knn_acu_train_media = []
knn_acu_train_std = []
knn_acu_test_media = []
knn_acu_test_std = []
ks = [3, 5, 7, 9, 11]

for i in ks:
    teste = []
    treino = []
    for treino_ind, teste_ind in kf:
        treino_ind = list(treino_ind)
        teste_ind = list(teste_ind)
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(valores.iloc[treino_ind, :], classes.iloc[treino_ind])
        teste.append(accuracy_score(classes.iloc[teste_ind], knn.predict(valores.iloc[teste_ind, :])))
        treino.append(accuracy_score(classes.iloc[treino_ind], knn.predict(valores.iloc[treino_ind, :])))
    knn_acu_test_media.append(np.mean(teste))
    knn_acu_train_media.append(np.mean(treino))
    knn_acu_test_std.append(np.std(teste))
    knn_acu_train_std.append(np.std(treino))
    if max(knn_acu_test_media)<knn_acu_test_media[-1] or i==3 :
        tabela[0]=teste

    

plt.clf()
plt.scatter(ks, knn_acu_test_media, c="red")
plt.plot(ks, knn_acu_test_media, c="red")
plt.scatter(ks, knn_acu_train_media, c="blue")
plt.plot(ks, knn_acu_train_media, c="blue")
plt.legend( ['Teste', 'Treino'])
plt.title('Diabetes')
plt.ylabel('Acuracia')
plt.xlabel('Numero de vizinhos mais proximos')
plt.savefig('knn.png')

acuracias_media.append(max(knn_acu_test_media))
acuracias_std.append(max(knn_acu_test_std))
k_usado = ks[knn_acu_test_media.index(max(knn_acu_test_media))]

# classificador naive bayes
teste = []
for treino_ind, teste_ind in kf:
    treino_ind = list(treino_ind)
    teste_ind = list(teste_ind)
    gnb = GaussianNB()
    gnb.fit(valores.iloc[treino_ind, :], classes.iloc[treino_ind])
    teste.append(accuracy_score(classes.iloc[teste_ind], gnb.predict(valores.iloc[teste_ind, :])))

acuracias_media.append(np.mean(teste))
acuracias_std.append(np.std(teste))
tabela.append(teste)

# classificador Decision Tree
teste = []
for treino_ind, teste_ind in kf:
    treino_ind = list(treino_ind)
    teste_ind = list(teste_ind)
    clf = tree.DecisionTreeClassifier()
    clf.fit(valores.iloc[treino_ind, :], classes.iloc[treino_ind])
    teste.append(accuracy_score(classes.iloc[teste_ind], clf.predict(valores.iloc[teste_ind, :])))


tree.export_graphviz(clf, out_file='arvore.dot')
# Usar o comando abaixo para converter a saida em um arquivo postscript
os.system("dot -Tps arvore.dot -o arvore.eps")
os.system("rm arvore.dot")

acuracias_media.append(np.mean(teste))
acuracias_std.append(np.std(teste))
tabela.append(teste)

plt.clf()
plt.bar([1, 2, 3], acuracias_media,color='r', yerr=acuracias_std)
plt.xlim(0.5, 4.5)
plt.ylim(0, 1.04)
plt.xticks([1.4, 2.4, 3.4], ["Knn", "Naive Bayes", "Decision Tree"])
plt.ylabel('Probabilidade')
plt.xlabel('Algoritimos')
plt.title('Diabetes')
plt.savefig('acuracias.png')

tabela=map(list, zip(*tabela))

df = pd.DataFrame(tabela, columns=["Knn","Naive Bayes","Decision tree"])
df.to_csv('tabela.csv', header=True, sep=',')


print "Fim l"
