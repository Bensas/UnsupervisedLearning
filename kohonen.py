import numpy as np
class Kohonen():
  def __init__(self, dimension, data, learning_rate):
    self.neurons = np.zeros((dimension, dimension, data.shape[1]))
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
  
  def get_neighbor_neurons(self, neuron):
    result = []
    neuron_x = neuron[0]
    neuron_y = neuron[1]
    #up
    if neuron_exists(neuron_x, neuron_y-1):
      result.append((neuron_x, neuron_y-1))
    #right
    if neuron_exists(neuron_x+1, neuron_y):
      result.append((neuron_x+1, neuron_y))
    #bottom
    if neuron_exists(neuron_x, neuron_y+1):
      result.append((neuron_x, neuron_y+1))
    #left
    if neuron_exists(neuron_x-1, neuron_y):
      result.append((neuron_x-1, neuron_y))
    return result
  
  def neuron_exists(neuron):
    return neuron[0] < self.dimensions && neuron[0] >= 0
          && neuron[1] < self.dimensions && neuron[1] >= 0


  def train(self, data, epochs=1000, learning_rate_decay=0.):
    for _ in range(epochs):
      for point in data:
        closest_neuron = self.get_closest_neuron(point)
        neighbor_neurons = self.get_neighbor_neurons(closest_neuron)
        # self.learning_rate = self.learning_rate - learning_rate_decay 
     