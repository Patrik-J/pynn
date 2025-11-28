from .sigmoid import sigmoid, dv_sigmoid
import numpy as np

class Layer:
    """
        Class for a layer holding N neurons.
    """
    def __init__(self, inputs:int, neurons:int, learning_rate:float, activation:callable, dv_activation:callable):
        """
            Create a layer instance.
            
            Parameters
            -------------------
                inputs : int
                    Number of inputs per neuron.
                neurons : int
                    Number of neurons in the layer.
                learning_rate : float 
                    The learning rate during the backpropagation.
                activation : callable
                    The activation function for this layer.
                dv_activation : callable
                    The derivative of the used activation function.
        """
        
        
        # initialize weights matrix with random values
        self.weights = np.random.randn(neurons, inputs) * np.sqrt(1 / inputs)
        
        # initialize bias vector with random values 
        self.bias = np.random.rand(neurons, 1) 
        
        # store learning rate
        self.lr = learning_rate

        # store activation function
        self.activation = activation 
        self.dv_activation = dv_activation

    def inject(self, a:np.ndarray) -> np.ndarray:
        """
            Forward propagation.
            
            Parameters
            ----------------
                a : np.ndarray
                    The activation vector of the previous layer.
                    
            Returns
            ----------------
                np.ndarray
                    The activation vector of the current layer.
        """
        
        # store the input, which is the output of the previous layer
        self.previous_a = a
        
        # determine the output of this layer
        self.z = self.weights @ a + self.bias        # z = Wx + b
        self.a = self.activation(self.z)        # output of each neuron a = σ(z) (elementwise; σ is the activation function, e.g. the sigmoid function)
        
        # return output
        return self.a
    
    def update(self, delta:np.ndarray) -> np.ndarray: 
        """  
            Backwards propagation.
            
            Parameters
            -------------------
                delta : np.ndarray
                    Delta for the current layer.
        
            Returns
            -------------------
                np.ndarray
                    New delta for next prior layer.
        """
        # get delta for next down layer
        new_delta = (self.weights.T @ delta) * self.previous_a * (1 - self.previous_a)
        
        # update weights
        self.weights -= self.lr * (delta @ self.previous_a.T)
        
        # update bias
        self.bias -= self.lr * delta 
        
        # return new delta
        return new_delta
        
    def get_weights_and_bias(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.weights, self.bias)