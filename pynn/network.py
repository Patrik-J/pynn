from .layer import Layer
from .sigmoid import sigmoid, dv_sigmoid
from .relu import RELU, dv_RELU
from .softmax import softmax, dv_softmax
from .training import TrainingContainer, TrainingExample
from .backpropagation import backprop
from .gradient import initial_delta
from .save import store_neural_network
from .load import load_NN

import numpy as np

class Network:
    """
        Class for holding multiple layers of neurons, therefore achieving a neural network. The hidden layers use the sigmoid/RELU function, while the final output layer uses the softmax function.
    """
    
    def __init__(self, layers:int, initial_inputs:int, final_outputs:int, learning_rate:float | list[float], neurons_per_layer:int, hidden_activation:str = "sigmoid") -> None:
        """ 
            Initialize the neural network and its layers.
            
            Parameters
            ----------------
                layers : int
                    The number of layers to be implemented.
                initial_inputs : int
                    The number of initial inputs.
                final_outputs : int
                    The number of final outpus.
                learning_rate : float | list[float]
                    The learning rate during the gradient descent.
                neurons_per_layer : int
                    The number of neurons per layer within the hidden layers.
                hidden_activation : str
                    The chosen activation function for the hidden layers. Options: 'sigmoid', 'relu'
        """
        
        # check number of given layers
        if layers < 2:
            raise Exception(f"Number of layers must not be smaller than 2 (current: {layers})!")
        else:
            if type(neurons_per_layer) == list:
                if len(neurons_per_layer) != layers:
                    raise Exception("Unequal number of layers and neurons per layer given!")
            if type(learning_rate) == list:
                if len(learning_rate) != layers:
                    raise Exception("Unequal number of layers and learning rates given!")

        self.activation_id = hidden_activation

        # store activation
        match hidden_activation:
            case "sigmoid":
                self.activation = sigmoid
                self.dv_activation = dv_sigmoid
            case "relu":
                self.activation = RELU
                self.dv_activation = dv_RELU
            case _:
                raise Exception(f"Unknown option {hidden_activation} for activation function of the hidden layers.")
        
        # store number of layers
        self.num_layers = layers
        
        # store number of inputs and outputs
        self.inputs = initial_inputs
        self.outputs = final_outputs
        
        # store learning rate
        self.lr = learning_rate
        
        # store number of neurons per layer
        self.neurons_per_layer = neurons_per_layer
        
        # initialize list
        self.layers:list[Layer] = self.__create__layers__()
        
        # initialize training container
        self.training_container:TrainingContainer = TrainingContainer()
        
        # reset cost
        self.cost = ""
        
    def __create__layers__(self) -> list[Layer]:
        lr = []
        if type(self.lr) == float:
            for _ in range(self.num_layers):
                lr.append(self.lr)
        else: 
            lr = self.lr    
 
        layers:list[Layer] = list()
        
        # add hidden layers
        for i in range(self.num_layers - 1):
            layer = Layer(inputs=inputs, neurons=self.neurons_per_layer, learning_rate=lr[i], activation=self.activation, dv_activation=self.dv_activation)
            inputs = self.neurons_per_layer
            layers.append(layer)
        
        
        # add final output layer
        out = Layer(inputs=inputs, neurons=self.outputs, learning_rate=lr[-1], activation=softmax, dv_activation=dv_softmax)
        layers.append(out)
        
        return layers          
    
    def __forward_propagation__(self, te:TrainingExample) -> np.ndarray:
        nn_output = te.get_nn_input()
        for i in range(self.num_layers):
            current_layer:Layer = self.layers[i]
            nn_output = current_layer.inject(nn_output)
        return nn_output
        
    def __backward_propagation__(self) -> None:
        delta = initial_delta(self.training_container, cost=self.cost)
        backprop(self.layers, delta)
    
    def __load_weights_and_bias__(self, directory:str) -> bool:
        """
            Load the weights and bias from save files located in the given directory.
            
            Parameters
            ---------------
                directory : str
                    Target directory
            
            Returns
            ---------------
                bool 
                    Successfully loaded weights and biases to neural network?
        """    
    
    def ask(self, input_value:np.ndarray) -> np.ndarray:
        """
            Give the nerual network an input vector and return the output of the final layer.
            
            Parameters
            ---------------
                input_value : np.ndarray
                    Input vector
            
            Returns
            ---------------
                np.ndarray
                    Output vector
        """
        
        nn_output = input_value 
        for i in range(self.num_layers):
            current_layer:Layer = self.layers[i]
            nn_output = current_layer.inject(nn_output)
        return nn_output
    
    def __return_layers__(self) -> list[Layer]:
        """
            Return list of layers to retrieve the weights and biases.
        """
        return self.layers
    
    def save(self, directory:str) -> bool:
        """
            Save all important parameters of the neural network to load them at a later point.
            
            Parameters
            ------------
                directory : str
                    Target directory
                    
            Returns
            -------------
                bool
                    Successfully saved?
        """
        return store_neural_network(nn=self, directory=directory)
    
    def __set_weights_and_bias__(self, index:int, weights:np.ndarray, bias:np.ndarray) -> None:
        self.layers[index].weights = weights
        self.layers[index].bias = bias
    
    @staticmethod
    def load(directory:str) -> "Network":
        load_NN(directory=directory)