import numpy as np
from .training import TrainingContainer
from .cost import gradient

def initial_delta(tc:TrainingContainer, fprime:callable, cost:str) -> np.ndarray:
    """ 
        Determine the initial delta δ^L by determining it for each training example and taking the average.
    """
    cost_gradient = gradient(cost)
    
    delta = np.zeros_like(tc[0][2], dtype=float)
    for example in tc:
        nn_output = example.get_neural_output()
        target = example.get_target_value()
        
        delta += cost_gradient(nn_output, target)
        
    return 1/tc.get_number_of_examples() * delta        