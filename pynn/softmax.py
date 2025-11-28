import numpy as np

def softmax(x:np.ndarray) -> np.ndarray:
    exp_sum = 0
    for i in range(len(x)):
        exp_sum += np.exp(x[i])
        
    return np.exp(x)/exp_sum

def dv_softmax(x:np.ndarray) -> np.ndarray:
    return softmax(x) * (1 - softmax(x))