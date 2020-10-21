import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from kohonen import Kohonen
from hopfield import Hopfield

KOHONEN_DIMENSION = 4

def custom_fmt(spec):
  return 'Countries'

command = input("Select the desired excercise:")

if command == "1a":

  print("Loading data...")
  raw_data = pd.read_csv("europe.csv")
  country_names = raw_data["Country"]
  raw_data = raw_data.drop(columns="Country")
  data_normalized = (raw_data - raw_data.mean()) / raw_data.std()
  data = data_normalized.to_numpy()
  # for row in data:
  #   print(row)

  print("Data loaded and normalized. Training net...")
  kohonen_net = Kohonen(dimension=KOHONEN_DIMENSION, data=data, learning_rate=0.001)
  kohonen_net.train(data=data)
  # print(kohonen_net.neurons)

  print("Net trained, obtaining results:")
  
  neuron_countries = []
  for i in range(KOHONEN_DIMENSION):
    row = []
    for j in range(KOHONEN_DIMENSION):
      row.append('(' + str(i) + ', ' + str(j) + ')')
    neuron_countries.append(row)

  for country_name, country_weights in zip(country_names, data):
    neuron = kohonen_net.get_closest_neuron(country_weights)
    neuron_countries[neuron[1]][neuron[0]] += country_name + '\n'
  
  for i in range(len(neuron_countries)):
    for j in range(len(neuron_countries[0])):
      plt.text(i, j, neuron_countries[i][j])
  plt.xlim(0, len(neuron_countries)*1.2)
  plt.ylim(0, len(neuron_countries)*1.2)
  plt.show()
  print(neuron_countries)
  # kohonen_net.plot_average_distances()
  kohonen_net.plot_average_distances_with_countries(neuron_countries=neuron_countries)
  print("Exiting.")

elif command == "2":
  print("Loading data...")
  raw_data = pd.read_csv("letters-flattened.csv")
  data = raw_data.to_numpy()
  hopfield_net = Hopfield(data)
  hopfield_net.train(data)
#LEE EL COMENTARIO JUAN -------------------------------------------------------------->
  test_patterns, expected_result_indexes = Hopfield.machine_of_patterns(data, 0.2, 50) #los dos ultimos argumentos van al reves, el ultimo es la probabilidad y el ante ultimo la cant de nuevos patterns
  for index, pattern in enumerate(test_patterns):
    stabilized_pattern = hop
  