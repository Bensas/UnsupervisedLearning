import pandas as pd

from kohonen import Kohonen

command = input("Select the desired excercise:")

if command == "1a":
  raw_data = pd.read_csv("data/europe.csv")
  raw_data = raw_data.drop(columns='Country')
  data_normalized = (raw_data - raw_data.mean()) / raw_data.std()
  data = data_normalized.to_numpy()
  kohonen_net = Kohonen(width=10, height=10, data=data)
