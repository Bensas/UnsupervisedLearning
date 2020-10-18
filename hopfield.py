import numpy as np

class Hopfield():
  def __init__(self, data):
    self.dimension = data.shape[1]
    self.weights = np.zeros((self.dimension, self.dimension))

  def train(self, data):
    for input_elem in data:
      for i in range(self.dimension):
        for j in range(self.dimension):
          if i != j:
            self.weights[i, j] += (1/self.dimension) * (input_elem[i] * input_elem[j])
    print(self.weights)
  
            


