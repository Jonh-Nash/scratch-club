import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    if x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x)
        x /= np.sum(x)
