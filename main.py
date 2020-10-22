import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from kohonen import Kohonen
from hopfield import Hopfield
from ojarule import ej1_b

KOHONEN_DIMENSION = 8
KOHONEN_INITIAL_RADIUS = 4

HOPFIELD_NUM_OF_TEST_PATTERNS = 500

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

  print("Data loaded and normalized. Training net...")
  kohonen_net = Kohonen(dimension=KOHONEN_DIMENSION, data=data, learning_rate=0.5, initial_radius=KOHONEN_INITIAL_RADIUS)
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

elif command == "1b":
  ej1_b()

elif command == "2":
  print("Loading data...")
  raw_data = pd.read_csv("letters-flattened.csv")
  data = raw_data.to_numpy()
  hopfield_net = Hopfield(data)
  hopfield_net.train(data)

  test_patterns, expected_result_indexes = Hopfield.machine_of_patterns(data, 0.2, HOPFIELD_NUM_OF_TEST_PATTERNS)
  correct_results = 0
  for index, pattern in enumerate(test_patterns):
    stabilized_pattern = hopfield_net.stabilize_pattern(pattern)
    if (data[expected_result_indexes[index]] == stabilized_pattern).all():
      correct_results += 1;

  print('Correct stabilizations: ' + str((correct_results / HOPFIELD_NUM_OF_TEST_PATTERNS) * 100) + '%')

  