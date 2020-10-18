import numpy as np
class Kohonen():
  def __init__(self, dimension, data, learning_rate):
    self.neurons = np.random.uniform(low=-1, high=1, size=(dimension, dimension, data.shape[1]))
    self.learning_rate = learning_rate
    self.dimension = dimension
  
  def get_closest_neuron(self, point):
    min_dist = np.linalg.norm(point - self.neurons[0][0])
    min_dist_neuron = (0, 0)
    for ix in range(self.dimension):
      for iy in range(self.dimension):
        dist = np.linalg.norm(point - self.neurons[iy, ix])
        if dist < min_dist:
          min_dist = dist
          min_dist_neuron = (ix, iy)
    return min_dist_neuron
  
  def get_neighbor_neurons(self, neuron, distance):
    result = []
    for ix in range(self.dimension):
      for iy in range(self.dimension):
        if neuron != (ix, iy) and self.dist(neuron, (ix, iy)) <= min_dist:
          result.append((ix, iy))
    return result
  
  def dist(self, X1, X2):
    return ((X2[0]-X1[0])**2 + (X2[1]-X1[1])**2)**(1/2)
  
  def neuron_exists(self, neuron):
    return neuron[0] < self.dimensions and neuron[0] >= 0 and neuron[1] < self.dimensions and neuron[1] >= 0


  def train(self, data, epochs=1000, learning_rate_decay=0):
    for _ in range(epochs):
      for point in data:
        closest_neuron = self.get_closest_neuron(point)
        closest_neuron_weights = self.neurons[closest_neuron[1], closest_neuron[0]]
        neighbor_neurons = self.get_neighbor_neurons(closest_neuron)
        for neuron in neighbor_neurons:
          self.neurons[neuron[1], neuron[0]] += self.learning_rate * (closest_neuron_weights - self.neurons[neuron[1], neuron[0]])
        self.learning_rate = self.learning_rate - learning_rate_decay 
