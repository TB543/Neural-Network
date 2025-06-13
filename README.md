# NeuralNetwork

A simple Python implementation of a fully connected feedforward neural network with sigmoid activation, supporting forward propagation and backpropagation learning.

## Features

- Define network architecture by specifying number of neurons in each layer.
- Forward pass to compute outputs from inputs.
- Backpropagation algorithm for training with adjustable learning rate.
- Save and load network state for reuse.
- Uses sigmoid activation function.
- Error checking on inputs and expected outputs.

## Installation

This project requires **Python** and **NumPy**.

Install NumPy if you haven't already:

```bash
pip install numpy
```

## Usage

```python
import NeuralNetwork

# Create a network with 3 input neurons, 5 neurons in one hidden layer, and 2 output neurons
nn = NeuralNetwork([3, 5, 2])

# Example input (values between 0 and 1)
inputs = [0.1, 0.5, 0.9]

# Compute the output
outputs = nn.get_outputs(inputs)
print("Outputs:", outputs)

# Expected output for training
expected = [0.0, 1.0]

# Train the network via backpropagation with default learning rate 0.05
cost = nn.back_propagate(expected)
print("Cost:", cost)

# Save the network state to a string
network_data = nn.save()

# Load a network from saved data string
nn2 = NeuralNetwork(network_data)
