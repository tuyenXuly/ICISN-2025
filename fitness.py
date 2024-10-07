import numpy as np

def count_points_in_circle(x_s, y_s,sensing_range,matrix,width,length):
    x_min = max(0, int(x_s - sensing_range))
    x_max = min(width, int(x_s + sensing_range) + 1)
    y_min = max(0, int(y_s - sensing_range))
    y_max = min(length, int(y_s + sensing_range) + 1)

    y_range = np.arange(y_min, y_max)
    x_range = np.arange(x_min, x_max)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    distances = np.sqrt((x_grid - x_s) ** 2 + (y_grid - y_s) ** 2)
    mask = distances <= sensing_range
    matrix_slice = matrix[y_min:y_max, x_min:x_max]  # Ensure correct slicing
    np.maximum(matrix_slice, mask, out=matrix_slice)  # Update matrix_slice with mask

def fitness_function(solution,M,sensing_range,area_width,area_length):
    matrix_width=area_length+1
    matrix_length=area_length+1
    matrix = np.zeros((matrix_width, matrix_length))
    for i in range(solution.shape[0]):
        x, y = solution[i, :]
        count_points_in_circle(x, y,sensing_range,matrix,area_width,area_length)
    count = np.sum(matrix == 1)
    return (count / M)*100