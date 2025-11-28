import numpy as np
from .layer import Layer

def backprop(layers:list[Layer], initital_delta:np.ndarray) -> None:
    delta = initital_delta
    for i in reversed(len(layers)):
        # get single layer
        layer:Layer = layers[i]
        
        # determine gradient of the cost function with respect to the weights and bias        
        weight_grad = delta @ layer.previous_a 
        bias_grad = delta
        
        # update delta 
        delta = (layer.weights.T @ delta) * layer.dv_activation(layer.previous_z)
        
        # learning rate
        lr = layer.lr
        
        # update weights and bias
        layer.weights -= lr*weight_grad
        layer.bias -= lr*bias_grad
    
        # store updated layer    
        layers[i] = layer