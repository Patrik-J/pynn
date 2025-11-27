import numpy as np 
from .training import TrainingContainer, TrainingExample

def single_cost_MSE(te:TrainingExample) -> float:
    """
        Uses mean-squared-error.
    """
    sum = 0
    diff = te.get_target() - te.get_input()
    for i in range(len(diff)):
        sum += diff[i]**2 
    return 0.5*sum    
    
def single_cost_CE(te:TrainingExample) -> float:
    """
        Uses cross-entropy error.
    """
    input, target = te.get_nn_input(), te.get_target_value()
    return - (target*np.log(input) + (1 - target)*np.log(1 - input))

def cost(container:TrainingContainer, function:str) -> float:
    """  
        Implemented cost function: C(x) = 1/n Σ_i C_i(x_i) where x_i are the single training examples.
    """    
    func = None 
    match function:
        case "mse":
            func = single_cost_MSE
        case "ce":
            func = single_cost_CE
        case _:
            raise Exception(f"Unknown option '{function}' for cost function.")
    sum = 0
    N = container.get_number_of_examples()
    for i in range(N):
        sum += func(container[i])
    return sum * 1/N