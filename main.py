from numpy import array as _vector, exp as _exp, set_printoptions as _set_printoptions, inf as _inf, log as _ln


class NeuralNetwork:
    """
    a neural network class for artificial learning

    todo:
     add way to initialize weights/biases to random values instead of 0
     add way to apply different activation functions
     add way to change range of possible values
     change learning rate based on cost to learn faster
    """

    # configures setting to save all of the arrays/vectors for the layers, weights and biases
    _set_printoptions(threshold=_inf)

    def __init__(self, layers: list or str) -> None:
        """
        creates neural network with all neurons, weights and biases set to 0

        :param layers: a list of ints where the length of the list is the number of layers and list contents is the
            number of neurons in each layer, if a string is given it should be the data from another network and it
            will be loaded
        """

        # loads previous neural network
        if isinstance(layers, str):
            self.load(layers)

        # creates new neural network
        elif isinstance(layers, list):

            # checks for errors
            if [type(layer) != int or layer <= 0 for layer in layers].count(True) > 0:
                raise ValueError('invalid list value given: list should only contain integers greater than 0')

            # creates network attributes if no errors are found
            self.layers = [_vector([0.0] * layer) for layer in layers]
            self.weights = [_vector([[0.0] * layers[index]] * layers[index + 1]) for index in range(len(layers) - 1)]
            self.biases = self.layers[1:]

        # raises error if invalid type is given
        else:
            raise TypeError('invalid variable type given should be string or list')

    def get_outputs(self, inputs: list) -> list:
        """
        calculates the outputs of the network

        :param inputs: a list of floats between 0 and 1 representing data values for the input (1st) layer

        :return: a list of floats between 0 and 1 representing the data values for the output (last) layer
        """

        # checks for errors
        if (not isinstance(inputs, list)) or \
                [isinstance(value, (float, int)) and 1 >= value >= 0 for value in inputs].count(False) > 0 or \
                len(inputs) != len(self.layers[0]):
            raise ValueError('invalid inputs given, make sure the length of inputs is equal to the length of the input '
                             '(first) layer of the network\nalso the inputs should only be values between 0 and 1')

        # calculates outputs
        self.layers[0] = _vector([float(value) for value in inputs])
        for index in range(len(self.layers) - 1):
            self.layers[index + 1] = _vector(list(map(
                self._sigmoid, self.weights[index].dot(self.layers[index]) + self.biases[index])))

        # returns final layer as a list
        return self.layers[-1].tolist()

    def back_propagate(self, expected_outputs: list, learning_rate: float = .05) -> float:
        """
        the way the network "learns" note: self.get_outputs must be run before back propagation is preformed

        :param expected_outputs: the expected output variables for output (last) layer of the network
        :param learning_rate: the pace at which the network learns, should be positive, default is .05

        :return: the cost function used for the calculations
        """

        # checks for errors
        if (not isinstance(expected_outputs, list)) or learning_rate < 0 or \
                [isinstance(value, (float, int)) and 1 >= value >= 0 for value in expected_outputs].count(False) > 0 \
                or len(expected_outputs) != len(self.layers[-1]):
            raise ValueError('invalid expected outputs given, make sure the length of inputs is equal to the length of '
                             'the output (last) layer of the network\nthe expected outputs should only be values '
                             'between 0 and 1\nadditionally the learning rate should only be positive')

        # calculates partial derivative for cost / weighted sum of output layer
        current_derivative = _vector([2 * (self.layers[-1][neuron_index] - expected_outputs[neuron_index]) *
                                      self._sigmoid_prime(self._inverse_sigmoid(self.layers[-1][neuron_index]))
                                      for neuron_index in range(len(self.layers[-1]))])

        # goes backwards through every layer
        for layer_index in reversed(range(len(self.layers) - 1)):

            # updates biases
            for neuron_index in range(len(self.biases[layer_index])):
                self.biases[layer_index][neuron_index] -= current_derivative[neuron_index] * learning_rate

            # updates weights
            old_weights = self.weights[layer_index].copy()
            for next_neuron_index in range(len(self.weights[layer_index])):
                for previous_neuron_index in range(len(self.weights[layer_index][next_neuron_index])):
                    self.weights[layer_index][next_neuron_index][previous_neuron_index] -= current_derivative[
                        next_neuron_index] * self.layers[layer_index][previous_neuron_index] * learning_rate

            # updates the partial derivative for the next layer to update
            current_derivative = current_derivative.dot(old_weights)
            for neuron_index in range(len(current_derivative)):
                current_derivative[neuron_index] = current_derivative[neuron_index] * \
                    self._sigmoid_prime(self._inverse_sigmoid(self.layers[layer_index][neuron_index]))

        # returns the cost of the network
        return sum([(self.layers[-1][neuron_index] - expected_outputs[neuron_index]) ** 2
                    for neuron_index in range(len(self.layers[-1]))])

    def save(self) -> str:
        """
        saves the data for the current network

        :return: a string representation of the json data of the network
        """

        return str(vars(self)).replace('array', '_vector')

    def load(self, data: str) -> None:
        """
        loads a previously saved neural network note: this will override current network

        :param data: the string representation of the json data of another network
        """

        # attempts to load saved data
        try:
            data = eval(data)
            self.layers = data['layers']
            self.weights = data['weights']
            self.biases = data['biases']

        # raises error if failure
        except Exception:
            raise TypeError('invalid data given, data should be the string that is given from self.save()')

    @staticmethod
    def _sigmoid(value: float) -> float:
        """
        the activation function for this neural network is the sigmoid function: 1/(1 + e^-x), this hidden method
        applies this function to the given value

        :param value: the weighted sum of the neuron

        :return: the activation value of the neuron
        """

        return 1 / (1 + _exp(-value))

    def _sigmoid_prime(self, value: float) -> float:
        """
        the derivative of the sigmoid function

        :param value: the value to plug into the function

        :return: the derivative of the sigmoid function
        """

        return self._sigmoid(value) * (1 - self._sigmoid(value))

    @staticmethod
    def _inverse_sigmoid(value: float) -> float:
        """
        undoes the sigmoid function to get the weighted sum during back propagation

        :param value: the activation value of the neuron

        :return: the weighted sum of the neuron
        """

        return _ln(value / (1 - value))
