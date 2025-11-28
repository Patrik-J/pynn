import os 
from .network import Network
from .layer import Layer

import numpy as np

class ExistingDirectoryError(Exception):
    def __init__(self, msg:str) -> None:
        super.__init__(msg)
    
class SavingError(Exception):
    def __init(self, msg:str) -> None:
        super.__init__(msg)

def directory_exists(directory:str) -> bool:
    return os.path.isdir(directory)

def create_directory(directory:str) -> bool:
    if not directory_exists(directory):
        os.makedir(directory)
    else:
        raise ExistingDirectoryError(f"Given directory '{directory}' already exists.")

def store_neural_network(nn:Network, directory:str) -> bool:
    try:
        create_directory(directory=directory)
        layer_directory = f"{directory}\\layers"
        
        create_header_file(nn=nn, directory=directory)
        
        create_directory(directory=layer_directory)
        
        layers = nn.layers
        
        for i in range(nn.num_layers):
            store_layer(directory=layer_directory, layer=layers[i], num=i+1)
        
        return True
        
    except Exception as e:
        print(f"Exception caught:\n{e}")
        return False
    
def store_layer(directory:str, layer:Layer, num:int) -> None:
    weights, bias = layer.get_weights_and_bias()
    
    directory = f"{directory}\\layer_{num}"
    create_directory(directory=directory)
    
    weights_file = f"{directory}\\weights.txt"
    bias_file = f"{directory}\\bias.txt"
    
    write_array(weights_file, weights)
    write_array(bias_file, bias)
    
    
def create_header_file(nn:Network, directory:str) -> None:
    s = f"Number of layers: {nn.num_layers}\n"
    s += f"Number of inputs: {nn.inputs}\n"
    s += f"Number of outputs: {nn.outputs}\n"
    s += f"Used activation function: {nn.activation_id}\n"
    s += f"Number of neurons in each hidden layer: {nn.neurons_per_layer}\n"
    s += f"Learning rates of each layer: "
    for i in range(len(nn.lr)):
        s += f"{nn.lr[i]}:" 

    filename = f"{directory}\\nn_header.txt"
    f = open(filename, "w")
    f.write(s)
    f.close()
    
def write_array(file:str, array:np.ndarray) -> None:
    shape = array.shape
    s = ""
    dimensions = len(shape)
    match dimensions:
        case 1:
            for i in range(shape[0]):
                s += f"{array[i]}\n"
        case 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    s += f"{array[i,j]}"
                    if j < shape[1] - j:
                        s+= ","
                s += "\n"
        case _:
            raise SavingError(f"Shape {shape} of array is not valid.")
    
    f = open(file, "w")
    f.write(s)
    f.close()