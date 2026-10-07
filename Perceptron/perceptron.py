import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

##1- leitura da base de dados
df = pd.read_csv("perceptron.txt")
#extração de colunas
x = df[["x", "y"]].values #coordenadas dos pontos
y = df["classe"].values #vetor com as classes 
classes = np.unique(y)
#plotar dataset
plt.figure(figsize = (7,7))

for c in classes: #aqui começa o loop que desenha cada classe com uma cor diferente (parte estética)
    pts = x[y == c] #seleciona todas as linhas x correspondentes a c
    plt.scatter(pts[:,0], pts[:,1], label = c)

plt.legend()
plt.title("Dataset fornecido (4 classes)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
##2- treino do One-vs-Rest
classifiers = {}

for c in classes: #aqui começa o loop que cria UM perceptron para cada classe
#criação dos labels binários
    y_binary = (y == c).astype(int)

    clf = Perceptron(max_iter = 1000, tol = 1e-3)
    clf.fit(x, y_binary)
    classifiers[c] = clf
##3- regiões de decisão
xx, yy = np.meshgrid(np.linspace(-1, 6.5, 400),
                     np.linspace(-1, 6.5, 400))

grid = np.c_[xx.ravel(), yy.ravel()]
#score de cada classificador OvR
scores = np.zeros((grid.shape[0], len(classes)))

for i, c in enumerate(classes):
    clf = classifiers[c]
    scores[:, i] = clf.decision_function(grid)

z = np.argmax(scores, axis=1).reshape(xx.shape)
##4- plotar resultados
plt.figure(figsize = (7,7))
plt.contourf(xx, yy, z, alpha = 0.3, levels = len(classes))

for c in classes: #plota novamente os pontos reais
    pts = x[y == c]
    plt.scatter(pts[:,0], pts[:,1], label = c)
#parte finais
plt.legend()
plt.title("Regiões de Decisão (Perceptron One-vs-Rest)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
