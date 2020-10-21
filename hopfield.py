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
  
  def stabilize_pattern(self, pattern, epochs):
    curr_pattern = pattern
    prev_pattern = np.array(curr_pattern)
    prev_prev_pattern = np.array(curr_pattern)

    for t in range(epochs):
        prev_prev_pattern = np.array(prev_pattern)
        prev_pattern = np.array(curr_pattern)            
        for i in range(self.dimension):
            sum = 0
            for j in range(self.dimension):
                sum += self.weights[i,j] * prev_pattern[j]
            curr_pattern[i] = sign(sum)

        if (t > 1 and (prev_prev_pattern == curr_pattern).all()):
            break

    return curr_pattern
