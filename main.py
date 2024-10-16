import random
import numpy as np
import matplotlib.pyplot as plt
from connectivity import check_connectivity
from fitness import fitness_function
from generate import generate_new_position
from template import show_sensor_matrix

# Set initial parameters
sensing_range = 10
communication_range = 20
num_sensor = 60
num_solutions = num_sensor

matrix_length = 101 # 101 points
matrix_width = 101  # 101 points

area_length = 100 #100m
area_width = 100 #100m

M =  matrix_length* matrix_width
matrix = np.zeros((matrix_length, matrix_width))

# Set parameters for ABC
num_employed_bees = num_solutions
num_onlooker_bees = num_solutions
max_iterations = 1000
limit = 50  # Reduced limit to increase exploration

# Initialize population
solution = np.zeros((num_sensor, 2))
x0, y0 = 50, 50
solution[0, :] = [x0, y0]
for i in range(1, num_sensor):
    x_prev, y_prev = solution[i - 1, :]
    x_new, y_new = generate_new_position(x_prev, y_prev, communication_range)
    solution[i, :] = [x_new, y_new]

food_sources = np.copy(solution)
best_fitness = fitness_function(food_sources,M,sensing_range,area_width,area_length)
no_improvement_counters = np.zeros(num_employed_bees)

# Main loop
for iteration in range(max_iterations):
    # Employed bees phase
    for i in range(1,num_employed_bees):
        mutant = np.copy(food_sources)
        available_sensors = list(range(1,num_sensor))
        available_sensors.remove(i)
        dimension = random.choice(available_sensors)  # Fixed to range (0, num_sensor-1)
        phi = 2 * np.random.uniform(-1, 1, 2) * (1 - no_improvement_counters[i] / max_iterations) ** 5
        mutant[i, :] += phi*(mutant[i,:]-mutant[dimension,:])

        mutant[i, 0] = np.clip(mutant[i, 0], 0, area_length)
        mutant[i, 1] = np.clip(mutant[i, 1], 0, area_width)

        mutant_fitness = fitness_function(mutant,M,sensing_range,area_width,area_length)
        if check_connectivity(mutant, communication_range) and mutant_fitness > best_fitness:
            food_sources = np.copy(mutant)
            best_fitness = mutant_fitness
            no_improvement_counters[i] = 0
        else:
            no_improvement_counters[i] += 1

    # Calculate probabilities
    F = np.zeros(num_employed_bees)
    for i in range(1,num_employed_bees):
        property = np.delete(food_sources, i, axis=0)
        F[i] = fitness_function(property,M,sensing_range,area_width,area_length)
    P = F / np.sum(F)
    # Onlooker bees phase
    for j in range(1,num_onlooker_bees):
        selected_food_source = np.random.choice(num_employed_bees, p=P)
                                                                                                                                    
        mutant = np.copy(food_sources)
        available_sensors = list(range(1,num_sensor))
        available_sensors.remove(j)
        dimension = random.choice(available_sensors)  # Fixed to range (0, num_sensor-1)
        phi = 2 * np.random.uniform(-1, 1, 2) * (1 - no_improvement_counters[j] / max_iterations) ** 5
        mutant[selected_food_source, :] += phi*(mutant[j,:]-mutant[dimension,:])

        mutant[selected_food_source, 0] = np.clip(mutant[selected_food_source, 0], 0, area_length)
        mutant[selected_food_source, 1] = np.clip(mutant[selected_food_source, 1], 0, area_width)

        mutant_fitness = fitness_function(mutant,M,sensing_range,area_width,area_length)
        if check_connectivity(mutant, communication_range) and mutant_fitness > best_fitness:
            food_sources[selected_food_source] = mutant[selected_food_source]
            best_fitness = mutant_fitness
            no_improvement_counters[selected_food_source] = 0
        else:
            no_improvement_counters[selected_food_source] += 1

    # Scout bees phase
    for k in range(1,num_employed_bees):
        if no_improvement_counters[k] > limit:
            new_solution = np.zeros((num_sensor, 2))
            x0, y0 = 50, 50
            new_solution[0, :] = [x0, y0]
            for i in range(1, num_sensor):
                x_prev, y_prev = new_solution[i - 1, :]
                x_new, y_new = generate_new_position(x_prev, y_prev, communication_range)
                new_solution[i, :] = [x_new, y_new]
            mutant_fitness = fitness_function(new_solution,M,sensing_range,area_width,area_length)
            if mutant_fitness > best_fitness and check_connectivity(new_solution, communication_range):
                food_sources = np.copy(new_solution)
                best_fitness = mutant_fitness
                no_improvement_counters[k] = 0
            else:
                no_improvement_counters[k] += 1

    # Display best fitness for each iteration
    print(f"Iteration {iteration}, Best Fitness: { best_fitness}")

# Find the overall best solution
overall_best_solution = food_sources

show_sensor_matrix(num_sensor,sensing_range,overall_best_solution,best_fitness)

print(f"Overall Best Fitness: {best_fitness}")
