import numpy as np

class TrainingExample:
    """
        Class for holding a single training example.
    """
    
    def __init__(self, nn_input:np.ndarray, target:np.ndarray) -> None:
        """
            Paramters
            -------------
                nn_input : np.ndarray
                    The input for the neural network.
                target : np.ndarray
                    The target value for the output of the neural network.
            Both arguments must have shape (N,1).
        """
        
        self.nn_input = nn_input
        self.target_value = target 
        
    def get_nn_input(self) -> np.ndarray:
        return self.nn_input
    
    def get_target_value(self) -> np.ndarray:
        return self.target_value
        
    def __getitem__(self, index) -> np.ndarray:
        if index == 0:
            return self.get_nn_input()
        elif index == 1:
            return self.get_target_value()
        else:
            raise IndexError
    
class TrainingContainer:
    """ 
        Class for holding multiple training examples.
    """        
    
    def __init__(self) -> None:
        self.container = list()
        self.num_examples = 0

    def add_training_example(self, te:TrainingExample) -> None: 
        self.container.append(te) 
        self.num_examples += 1
    
    def get_training_example(self, index:int) -> TrainingExample:
        return self.container[index]
    
    def get_number_of_examples(self) -> int:
        return self.num_examples
    
    def reset_container(self) -> None:
        self.container = list()
        self.num_examples = 0
        
    def __iter__(self):
        self.index = 0
        return self 
    
    def __next__(self) -> TrainingExample:
        if self.index >= self.num_examples:
            raise StopIteration
        value = self.container[self.index]
        self.index += 1
        return value 
    
    def __getitem__(self, index) -> TrainingExample:
        return self.container[index]
    
    def __setitem__(self, index, value) -> None:
        self.container[index] = value 