import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

class Kohonen():
  def __init__(self, dimension, data, learning_rate):
    self.neurons = np.random.uniform(low=-1, high=1, size=(dimension, dimension, data.shape[1]))
    self.initial_learning_rate = learning_rate
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
        if neuron != (ix, iy) and self.dist(neuron, (ix, iy)) <= distance:
          result.append((ix, iy))
    return result
  
  #This function returns a matrix with the average distance form each neuron to al its neighbors
  def get_average_distances(self, neighbor_distance):
    average_distances = np.zeros((self.dimension, self.dimension))
    for ix in range(self.dimension):
      for iy in range(self.dimension):
        current_neuron = self.neurons[iy, ix]
        neighbor_neurons = self.get_neighbor_neurons(current_neuron, neighbor_distance)
        dist_sum = 0
        for neighbor_neuron in neighbor_neurons:
          neighbor = self.neurons[neighbor_neuron[1], neighbor_neuron[0]]
          dist_sum += np.linalg.norm(neighbor - current_neuron)
        average_distances[iy, ix] = dist_sum/len(neighbor_neurons)
    return average_distances

  def dist(self, X1, X2):
    return ((X2[0]-X1[0])**2 + (X2[1]-X1[1])**2)**(1/2)
  
  def neuron_exists(self, neuron):
    return neuron[0] < self.dimension and neuron[0] >= 0 and neuron[1] < self.dimension and neuron[1] >= 0

  def train(self, data, epochs=1000):
    for _ in range(epochs):
      learning_rate = self.initial_learning_rate 
      for index, point in enumerate(data):
        closest_neuron = self.get_closest_neuron(point)
        closest_neuron_weights = self.neurons[closest_neuron[1], closest_neuron[0]]
        neighbor_neurons = self.get_neighbor_neurons(closest_neuron, 1)
        for neuron in neighbor_neurons:
          self.neurons[neuron[1], neuron[0]] += learning_rate * (closest_neuron_weights - self.neurons[neuron[1], neuron[0]])
        learning_rate = learning_rate / (index+1)
  
  #Plotting
  def plot_average_distances(self, colormap='Blues', internal=False):
    neuron_average_distances = self.get_average_distances(neighbor_distance=1)
    ax = sns.heatmap(neuron_average_distances, cmap=colormap, annot=True)

    plt.title("Average Distance to Neighbors", fontsize=25)
    plt.show()
  
  def plot_average_distances_with_countries(self, neuron_countries, colormap='Blues', internal=False):
    neuron_average_distances = self.get_average_distances(neighbor_distance=1)
    akws = {"ha": 'left',"va": 'top'}
    ax = sns.heatmap(neuron_average_distances, cmap=colormap, annot=neuron_countries, annot_kws=akws, fmt="s")
    plt.title("Average Distance to Neighbors", fontsize=25)
    plt.show()