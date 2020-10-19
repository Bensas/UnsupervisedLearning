import sys
import numpy as np
import pandas as pd
import math
import time     
from numpy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class red_neuronal_with_oja(): #perceptron simple
    def __init__(self, no_of_inputs, values, epochs, learning_rate=0.01):
        self.values = values
        self.learning_rate = learning_rate
        self.variables = values.shape[1]
        weights = np.random.randint(0,50,self.variables)
        self.weights = weights / np.linalg.norm(weights, 2)
        self.bias = 0.01
        self.epochs = epochs
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        if summation >= 0:
          return 1
        else:
          return -1

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
    # countries = europe_csv.loc[:, 'Country'] #separo valores para trabajr mas facil luego, en registros y variables.
    values = europe_csv.loc[1:, ['Area','GPD','Inflation','Life.expect', 'Military', 'Pop.growth', 'Unemployment']].values.astype(np.float)
    
    print("Raw Values: \n")
    print(values)

    values_normalized = StandardScaler().fit_transform(values)

    print("Normalized Values: \n")
    print(values_normalized)

    # Let's check whether the normalized data has a mean of zero and a standard deviation of one.

    print("Mean: \n")
    print(np.mean(values_normalized))
    print("Standard deviation: \n")
    print(np.std(values_normalized))

    # With PCA

    print("\n\n\n WITH PCA \n\n\n")
    start = time.time()
    feat_cols = ['feature'+str(i) for i in range(values_normalized.shape[1])]
    normalised_values = pd.DataFrame(values_normalized,columns=feat_cols)
    normalised_values.tail()
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(values_normalized)
    principal_Df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    principal_Df.tail()
    end = time.time()

    #With Oja's rule

    epochs = 1000
    oja = red_neuronal_with_oja(values_normalized[0].size, values_normalized, epochs)
    start2 = time.time()
    oja.train()
    end2 = time.time()
    print(oja)

    if (end-start) > (end2-start2):
        print("Oja's rule was more efficient")
    else:
        print("PCA was more efficient")


if __name__ == '__main__':
    ej1_b()