import numpy as np 
from .training import TrainingContainer, TrainingExample

def single_cost(te:TrainingExample) -> float:
    sum = 0
    diff = te.get_target() - te.get_input()
    for i in range(len(diff)):
        sum += diff[i]**2 
    return 0.5*sum    
    
def cost(container:TrainingContainer) -> float:
    """  
        Implemented cost function: C(x) = 1/n Σ_i C_i(x_i) where x_i are the single training examples.
    """
    sum = 0
    N = container.get_number_of_examples()
    for i in range(N):
        sum += single_cost(container[i])
    return sum * 1/N