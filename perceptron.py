#!/usr/bin/python3
"""
ML Basic Perceptron basics
https://matt.might.net/articles/hello-perceptron/

Further reading
http://neuralnetworksanddeeplearning.com/
https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/


"""


def perceptron(inputs, weights, threshold):
    '''
    Simple perceptron
    '''
    weighted_sum = sum(x*w for x, w in zip(inputs, weights))
    return 1 if weighted_sum >= threshold else 0

def not_function(x):
    '''
    Perceptron not
    '''
    weight = -1
    threshold = -0.5
    return perceptron([x], [weight], threshold)


# Training

and_data = [
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1)
]

# The Perceptron algorithm means - finding a line (linear function), that
# separates between The values that return ones and those that return zeros
#
# The algo iteratively adjusts the weights of the linear function until we get
# the closest to the training data results.


"""
Learning algorithm:

1. Init the weights and threshold with random values
2. For each input-output pair in the training data:
* Compute teh perceptron output using the current weights and threshold
* Update the weights and threshold based on the difference between the desired
  output and the Perceptron output - the error.

Update rule:
* If the perceptron's output is correct, do not change the weights or threshold.
* If the perceptron's output is too low, increase the weights and decrease the threshold.
* If the perceptron's output is too high, decrease the weights and increase the threshold.

Learning rate - the size of the step on either direction
"""

import random

def train_perceptron(data, learning_rate=0.1, max_iter=1000):
    '''
    data - tuple of (inputs, output)
    '''
    first_pair = data[0]
    num_inputs = len(first_pair[0])

    # Initialize the vector of the weights and the threshold
    weights = [random.random() for _ in range(num_inputs)]
    threshold = random.random()

    for _ in range(max_iter):
        num_errors = 0  # How many wrong inputs for this iteration

        for inputs, desired_output in data:
            output = perceptron(inputs, weights, threshold)
            error = desired_output - output

            if error != 0:
                num_errors += 1
                for i in range(num_inputs):
                    weights[i] += learning_rate * error * inputs[i]
                threshold -= learning_rate * error
            if num_errors == 0:
                break

        # I think there's an bug here - keep the weights and threshold with
        # minimal error

    return weights, threshold

# Training:
and_weights, and_threshold = train_perceptron(and_data)
print(f'AND - weights: {and_weights}, Threshold: {and_threshold}')

or_data = [
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 1)
]

or_weights, or_threshold = train_perceptron(or_data)
print(f'OR - weights: {or_weights}, Threshold: {or_threshold}')

xor_data = [
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 0)
]

"""
Running Perceptron on XOR data won't work since there isn't a single line that
can separate the data into two groups
"""
xor_weights, xor_threshold = train_perceptron(xor_data, max_iter=10000)
print(f'XOR - weights: {xor_weights}, Threshold: {xor_threshold}')

print(perceptron((0, 0), xor_weights, xor_threshold))  # prints 0
print(perceptron((0, 1), xor_weights, xor_threshold))  # prints 1
print(perceptron((1, 0), xor_weights, xor_threshold))  # prints 0
print(perceptron((1, 1), xor_weights, xor_threshold))  # prints 1!!
