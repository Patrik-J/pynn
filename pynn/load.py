import numpy as np 
from .network import Network 
from .save import read_array

def __load_header(headerfile:str) -> tuple:
    """
        Load the data from a header file.
        
        Returns
        -----------------
            tuple[int, int, int, str, int, np.ndarray]
    """
    try:
        f = open(headerfile, "r")
        lines = f.readlines()
        f.close() 
            
        num_layers = int(lines[0].split(":")[1].removeprefix(" ").removesuffix("\n"))
        
        num_inputs = int(lines[1].split(":")[1].removeprefix(" ").removesuffix("\n"))
        
        num_outputs = int(lines[2].split(":")[1].removeprefix(" ").removesuffix("\n"))
        
        activation = str(lines[3].split(":")[1].removeprefix(" ").removesuffix("\n"))
        
        learning_rates = []
        
        lr = lines[3].split(":")[1].removeprefix(" ").removesuffix("\n").split(";")
        for i in range(len(lr)):
            learning_rates.append(float(lr[i]))
            
        learning_rates = np.array(learning_rates, dtype=float)
        
        return (num_layers, num_inputs, num_outputs, activation, learning_rates)
    
    except Exception as e:
        print(f"Exception caught:\n{e}\n\nDoes the header file exist?")
        raise Exception(e)
    
def __load_weights(layerdirectory:str) -> np.ndarray:
    weights_file = f"{layerdirectory}\\weights.txt"    
    weights = read_array(weights_file)
    return weights    
        
def __load_bias(layerdirectory:str) -> np.ndarray:
    bias_file = f"{layerdirectory}\\bias.txt"
    bias = read_array(bias_file)
    return bias
        
    
def load_NN(directory:str) -> Network:
    headerfile = f"{directory}\\nn_header.txt"
    
    num_layers, inputs, outputs, neurons, activation, lr = __load_header(headerfile=headerfile)
    
    nn = Network(layers=num_layers, initial_inputs=inputs, final_outputs=outputs, learning_rate=lr, neurons_per_layer=neurons, hidden_activation=activation)
    
    layer_directory = f"{directory}\\layers"
    
    for i in range(num_layers):
        layerpath = f"{layer_directory}\\layer_{i+1}"
        
        weights = __load_weights(layerdirectory=layerpath)
        
        bias = __load_bias(layerdirectory=layerpath)
        
        nn.__set_weights_and_bias__(i, weights, bias)
        
    return nn
    