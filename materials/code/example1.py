import numpy as np

def scoring_function(x, max_x, min_x):
    return 2 * (x - min_x) / (max_x - min_x) - 1

C = np.array([
    34.2, 
    65.4, 
    12.3, 
    32.7, 
    98.1, 
    7.4,
    2.6,
    120.9,
    12.4,
    56.0
])

max_x = np.max(C)
min_x = np.min(C)

S = scoring_function(C, max_x, min_x)
print(S)

