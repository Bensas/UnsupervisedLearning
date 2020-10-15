import numpy as np
class Kohonen():
  def __init__(self, width, height, data):
    self.neurons = np.zeros(height, width, data.shape[1])
  
  def get_closest_neuron(self, point):
    min_dist = np.linalg.norm(point - self.neurons[0][0])
    min_dist_neuron = (0, 0)
    for ix,iy in np.ndindex(self.neurons.shape):
      dist = np.linalg.norm(point - self.neurons[ix, iy])
      if dist < min_dist:
        min_dist = dist
        min_dist_neuron = (ix, iy)
    return min_dist


  def train(self, data, epochs=1000):
    for point in data:
      closest_neuron = self.get_closest_neuron(point)
     