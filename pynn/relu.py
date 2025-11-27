def RELU(x):
    if x > 0:
        return x 
    else:
        return 0

def dv_RELU(x):
    if x > 0:
        return 1
    else:
        return 0