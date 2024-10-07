import numpy as np

def generate_new_position(x_prev, y_prev, communication_range):
    r = np.random.uniform(5, communication_range)
    theta = np.random.uniform(0, 2 * np.pi)
    x_new = x_prev + r * np.cos(theta)
    y_new = y_prev + r * np.sin(theta)
    x_new = np.clip(x_new, 0, 100)
    y_new = np.clip(y_new, 0, 100)
    return x_new, y_new