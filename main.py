import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from kohonen import Kohonen
from hopfield import Hopfield

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
  kohonen_net = Kohonen(dimension=2, data=data, learning_rate=0.001)
  kohonen_net.train(data=data)
  # print(kohonen_net.neurons)

  print("Net trained, obtaining results:")
  
  neuron_countries = []
  for i in range(2):
    row = []
    for j in range(2):
      row.append('')
    neuron_countries.append(row)

  for country_name, country_weights in zip(country_names, data):
    neuron = kohonen_net.get_closest_neuron(country_weights)
    neuron_countries[neuron[1]][neuron[0]] += country_name + '\n'
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
  