import sys
import numpy as np
import pandas as pd
import math
import time     
import scipy.linalg as la
import matplotlib.pyplot as plt
from numpy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

class red_neuronal_with_oja(): #perceptron simple
    def __init__(self, values, epochs, learning_rate=0.01):
        self.values = values
        self.learning_rate = learning_rate
        self.variables = values.shape[1]
        weights = np.random.randint(0,50,self.variables)
        self.weights = weights / np.linalg.norm(weights, 2)
        self.bias = 0.01
        self.epochs = epochs
           
    def predict(self, sample):
        activ = 0
        for i in range(self.variables):
            activ += self.weights[i] * sample[i]
        return activ

    def train(self):
        for epoch in range(self.epochs):
            for sample in self.values:
                activ = self.predict(sample)
                for i in range(self.variables):
                    deltaW = self.learning_rate*activ*(sample[i] - (activ * self.weights[i]))
                    self.weights[i] = self.weights[i]+deltaW
            



def ej1_b():

    europe_csv = pd.read_csv('europe.csv', names=['Country','Area','GPD','Inflation','Life.expect', 'Military', 'Pop.growth', 'Unemployment'])
    europe_csv = europe_csv.drop([0]) #saco la primer fila de nombres ya que no la quiero en la matriz de datos
    countries = europe_csv.loc[:, 'Country'] #separo valores para trabajr mas facil luego, en registros y variables.
    values = europe_csv.loc[1:, ['Area','GPD','Inflation','Life.expect', 'Military', 'Pop.growth', 'Unemployment']].values.astype(np.float)

    print("Raw Values: \n")
    print(values)

    non_categoric = np.array(values)
    means = np.reshape(non_categoric.mean(axis=0), (-1, non_categoric.shape[1]))
    ones = np.reshape(np.ones(non_categoric.shape[0]), (-1, 1))
    mean_cero = non_categoric - (ones * means)
    
    values_normalized = normalize(mean_cero, axis=0)
    
    # METODO PARA NORMALIZAR ALTERNATIVO DUDOSO
    # scalar = MinMaxScaler(feature_range=(-1,1))
    # scalar.fit(values)
    # values_normalized = scalar.transform(values)

    print("Normalized Values: \n")
    print(values_normalized)

    # Let's check whether the normalized data has a mean of zero and a standard deviation of one.

    # print("Mean: \n")
    # print(np.mean(values_normalized))
    # print("Standard deviation: \n")
    # print(np.std(values_normalized))

    # With PCA

    print("\n\n\n WITH PCA \n\n\n")

    start = time.time()

    # feat_cols = [features[i] for i in range(values_normalized.shape[1])]
    # normalised_values = pd.DataFrame(values_normalized,columns=feat_cols)
    # print(normalised_values)

    eigenvalues, eigenvectors = np.linalg.eig(np.cov(values_normalized.T))
    # print(eigenvalues)

    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(values_normalized)
    principal_Df = pd.DataFrame(data = pca.components_.T, columns = ['Principal component 1'])

    end = time.time()

    print(principal_Df)
    # print("\n\n")
    # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    # plt.figure()
    # plt.figure(figsize=(10,10))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    # plt.xlabel('Principal Component - 1',fontsize=20)
    # plt.ylabel('Principal Component - 2',fontsize=20)
    # plt.title("Principal Component Analysis of Europe Dataset",fontsize=20)
    # for target in zip(countries):
    #     indicesToKeep = europe_csv['Country'] == target

    # FALLA ES LA LINEA 93!!!!!!!!!!!!!

    #     plt.scatter(principal_Df.loc[indicesToKeep, 'principal component 1'], principal_Df.loc[indicesToKeep, 'principal component 2'], c = 'r', s = 50)

    # plt.legend(countries,prop={'size': 15})

    #With Oja's rule

    print("\n\n\n WITH OJA'S RULE \n\n\n")

    epochs = 1000
    oja = red_neuronal_with_oja( values_normalized, epochs)
    start2 = time.time()
    oja.train()
    end2 = time.time()
    oja_Df = pd.DataFrame(data = oja.weights.T, columns = ['Principal component 1'])
    print(oja_Df)

    if (end-start) > (end2-start2):
        print("Oja's rule was more efficient")
    else:
        print("PCA was more efficient")


if __name__ == '__main__':
    ej1_b()
    