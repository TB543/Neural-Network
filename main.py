from numpy import array as _vector, exp as _exp, set_printoptions as _set_printoptions, inf as _inf, sum as _sum


class NeuralNetwork:
    """
    a neural network class for artificial learning
    """

    # configures setting to save all of the arrays/vectors for the layers, weights and biases
    _set_printoptions(threshold=_inf)

    def __init__(self, layers: list[int] or str):
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

    def get_outputs(self, inputs: list[float]):
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

    def back_propagate(self, expected_outputs: list[float], learning_rate: float = .05):
        """
        the way the network "learns" note: self.get_outputs must be run before back propagation is preformed

        :param expected_outputs: the expected output variables for output (last) layer of the network
        :param learning_rate: the pace at which the network learns, default is .05

        :return: the cost function used for the calculations
        """

        # checks for errors
        if (not isinstance(expected_outputs, list)) or \
                [isinstance(value, (float, int)) and 1 >= value >= 0 for value in expected_outputs].count(False) > 0 \
                or len(expected_outputs) != len(self.layers[-1]):
            raise ValueError('invalid expected outputs given, make sure the length of inputs is equal to the length of '
                             'the output (last) layer of the network\nalso the expected outputs should only be values '
                             'between 0 and 1')

        # calculates cost
        cost = (self.layers[-1] - _vector(expected_outputs)).sum() ** 2

        # performs backpropagation on output layer
        for neuron_index in range(len(self.layers[-1])):
            2 * ()

        # performs backpropagation on hidden layers
        for layer_index in range(len(self.layers)):
            pass

            # finds optimal change for weights

            # finds optimal change for biases

        """
        notes:
        
        d cost to neuron = 2(actual - expected)
        d neuron to weighted sum = neuron * (1 - neuron)
        d weighted sum to connection weight = previous neuron
        d weighted sum to neuron = weight
        d weighted sum to bias = 1
        
        """

        return cost

    def save(self):
        """
        saves the data for the current network

        :return: a string representation of the json data of the network
        """

        return str(vars(self)).replace('array', '_vector')

    def load(self, data: str):
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
    def _sigmoid(value: float):
        """
        the activation function for this neural network is the sigmoid function: 1/(1 + e^-x), this hidden method
        applies this function to the given value

        :param value: the weighted sum of the neuron

        :return: the weighted sum after the sigmoid is applied
        """

        return 1 / (1 + _exp(-value))
