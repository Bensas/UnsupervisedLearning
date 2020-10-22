import numpy as np
from random import random
from random import randint

def what(self, number):
    if(number > 0):
      return 1
    elif (number < 0):
      return -1
    else:
      return 0

class Hopfield():
  def __init__(self, data):
    self.dimension = data.shape[1]
    self.weights = np.zeros((self.dimension, self.dimension))

  def train(self, data):
    for input_elem in data:
      for i in range(self.dimension):
        for j in range(self.dimension):
          if i != j:
            self.weights[i, j] += (1/self.dimension) * (input_elem[i] * input_elem[j])
    print(self.weights)
  
  def stabilize_pattern(self, pattern, epochs=1000):
    curr_pattern = pattern
    prev_pattern = np.array(curr_pattern)
    prev_prev_pattern = np.array(curr_pattern)

    for t in range(epochs):
        prev_prev_pattern = np.array(prev_pattern)
        prev_pattern = np.array(curr_pattern)            
        for i in range(self.dimension):
            sum = 0
            for j in range(self.dimension):
                sum += self.weights[i,j] * prev_pattern[j]
            curr_pattern[i] = what(self, sum)

        if (t > 1 and (prev_prev_pattern == curr_pattern).all()):
            break

    return curr_pattern
  
  @classmethod
  def machine_of_patterns(cls, data, prob_of_mutation, n_patterns):
    #how many letters do we have
    amount_of_letters = data.shape[0]
    #length of an array of one letter
    length_of_letters = data.shape[1]
    #book = is all the new letters or patterns
    book = np.zeros((n_patterns, length_of_letters))
    #book_result = is what the new letters should be after evaluation
    book_result = np.zeros(n_patterns)
    for index in range(n_patterns):
        #letter to modify random
        chosen_letter = randint(0,amount_of_letters - 1)
        #We save the correct letter in the book of results
        book_result[index] = chosen_letter
        #We copy the letter also in the book that will later be modify
        book[index,:] = data[chosen_letter,:]
        for index2 in range(length_of_letters):
            #each number inside a letter has a chance of being modify
            if(random() < prob_of_mutation):
                book[index, index2] = book[index, index2] * (-1)
    return book, book_result.astype(int)