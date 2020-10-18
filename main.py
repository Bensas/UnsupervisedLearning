import pandas as pd

from kohonen import Kohonen

command = input("Select the desired excercise:")

if command == "1a":

  print("Loading data...")
  raw_data = pd.read_csv("europe.csv")
  country_names = raw_data["Country"]
  raw_data = raw_data.drop(columns="Country")
  data_normalized = (raw_data - raw_data.mean()) / raw_data.std()
  data = data_normalized.to_numpy()
  for row in data:
    print(row)

  print("Data loaded and normalized. Training net...")
  kohonen_net = Kohonen(dimension=2, data=data, learning_rate=0.001)

  print("Net trained, obtaining results:")
  lists = {
    "(0, 0)": [],
    "(0, 1)": [],
    "(1, 0)": [],
    "(1, 1)": []
  }
  for country_name, country_weights in zip(country_names, data):
    lists[str(kohonen_net.get_closest_neuron(country_weights))].append(country_name)
    print(country_name + " ---> " + str(kohonen_net.get_closest_neuron(country_weights)))
  for key in lists:
    print("Countries in " + key + ": ")
    for country in lists[key]:
      print(country)
  print("Exiting.")