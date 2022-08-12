from numpy import array as _vector


class NeuralNetwork:
    """
    a neural network class for artificial learning
    """

    def __init__(self, layers: list[int], saved_network: str = None):
        """
        creates neural network with all weights and biases set to 0

        :param layers: a list of ints where the length of the list is the number of layers and list contents is the
            number of neurons in each layer
        :param saved_network: an optional parameter, if given a previously saved network will be loaded
        """

    def get_outputs(self, inputs: list[float]):
        """
        calculates the outputs of the network

        :param inputs: a list of floats between 0 and 1 representing data values for the input (1st) layer

        :return: a list of floats between 0 and 1 representing the data values for the output (last) layer
        """

    def back_propagate(self, expected_outputs: list[float]):
        """
        the way the network "learns" note: self.get_outputs must be run before back propagation is preformed

        :param expected_outputs: the expected output variables for output (last) layer of the network
        """

    def save(self):
        """
        saves the data for the current network

        :return: a string representation of the json data of the network
        """

    def load(self, data: str):
        """
        loads a previously saved neural network note: this will override current network

        :param data: the string representation of the json data of another network
        """
