import numpy as np
from .layer import Layer

def backprop(layers:list[Layer], initital_delta:np.ndarray) -> None:
    delta = initital_delta
    for i in reversed(len(layers)):
        # get single layer
        layer:Layer = layers[i]
        
        # multiply delta with g'(z^l)
        if i > 0:
            delta *= layer.dv_activation(layer.z)
        
        # determine gradient of the cost function with respect to the weights and bias        
        weight_grad = delta @ layer.previous_a 
        bias_grad = delta
        
        # update delta (without multiplying by g'(z^l-1) just yet)
        # idea: δ^l-1 = W^T x δ^l * g'(z^l-1), but instead of storing z^l-1 and z^l in each layer, the multiplication with g'(z^l-1) is done in the next down layer, where it is simply g'(z^l)
        delta = (layer.weights.T @ delta)
        
        # learning rate
        lr = layer.lr
        
        # update weights and bias
        layer.weights -= lr*weight_grad
        layer.bias -= lr*bias_grad
    
        # store updated layer    
        layers[i] = layer