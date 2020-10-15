import pandas as pd

from kohonen import Kohonen

command = input("Select the desired excercise:")

if command == "1a":
  data = pd.read_csv("data/europe.csv")
  kohonen_net = Kohonen(width=10, height=10, data=data)
