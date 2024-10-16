from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
import math
import csv

# Set initial parameters
Rs=sensing_range = 10
Rc=communication_range = 10
nPop=num_sensor = 60
num_solutions = num_sensor
VarMaxx = length = 100
VarMaxy=width = 100
M = length * width
matrix = np.zeros((length, width))

# Set parameters for ABC
num_employed_bees = num_solutions
num_onlooker_bees = num_solutions
max_iterations = 2000
limit = 50  # Reduced limit to increase exploration

image = Image.open('C1_v2.png')
image_resized = image.resize((VarMaxx+1, VarMaxy+1))
image_L = image_resized.convert('L')
Area1 = np.zeros((VarMaxx+1,VarMaxy+1))
image_matrix = np.array(image_L)

image_1 =  Image.open('C1_real.png')

for i in range(VarMaxx+1):
    for j in range(VarMaxy+1):
        if image_matrix[i,j]  > 1:
            Area1[i,j] = 255
        else:
            Area1[i,j] = 1

ban_position_list = np.argwhere(Area1 == 1)
ban_position = [(x, y) for y, x in ban_position_list]

def fitness_function(sensor_nodes):
    M = (VarMaxx+1)*(VarMaxy+1)
    Rss = Rs ** 2 
    matrix_c = np.zeros((VarMaxx+1, VarMaxy+1), dtype=int)  
    grid_x, grid_y = np.meshgrid(np.arange(VarMaxx+1), np.arange(VarMaxy+1), indexing='ij')
    for sensor in sensor_nodes:
        sensor_y, sensor_x = sensor
        distances = (grid_x - sensor_x) ** 2 + (grid_y - sensor_y) ** 2
        matrix_c[distances <= Rss] = 1
    for i in ban_position:
        matrix_c[i[1],i[0]] = 0 
    coverage_ratio = round(np.sum(matrix_c) / M, 4)
    return coverage_ratio

def check_connectivity(positions, communication_range):
    dist_matrix = np.sqrt(np.sum((positions[:, np.newaxis] - positions) ** 2, axis=2))
    adjacency_matrix = (dist_matrix <= communication_range).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)
    graph = nx.Graph(adjacency_matrix)
    return nx.is_connected(graph)

def initialize_population(): 
    sink_node = (VarMaxx/2, VarMaxy/2)
    initPop = []
    initPop.append(sink_node)
    for i in range(1, nPop):
        check = True
        while check:
            xp, yp = initPop[np.random.randint(0,i)]
            Rcom = Rc*np.random.rand()
            xi = xp + 2*Rcom*np.random.rand()-Rcom
            if np.random.rand() >0.5:
                yi = yp + math.sqrt(Rcom**2 -(xi-xp)**2)
            else:
                yi = yp - math.sqrt(Rcom**2 -(xi-xp)**2)
            xi = np.clip(xi, 0, VarMaxx)
            yi = np.clip(yi, 0, VarMaxy)
            xj = int(xi)
            yj = int(yi)
            xj_c = xj+1
            yj_c = yj+1
            xj_t = xj-1
            yj_t = yj-1
            xj_c = np.clip(xj_c, 0, VarMaxx)
            yj_c = np.clip(yj_c, 0, VarMaxy)
            xj_t = np.clip(xj_t, 0, VarMaxx)
            yj_t = np.clip(yj_t, 0, VarMaxy)
            if (Area1[yj,xj] == 255
                and Area1[yj,xj_c] == 255 
                and Area1[yj,xj_t] == 255
                and Area1[yj_c,xj] == 255
                and Area1[yj_t,xj] == 255):
                check = False
        initPop.append((xi, yi))
    return initPop

food_sources = initialize_population()
best_fitness = fitness_function(food_sources)
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

        mutant[i, 0] = np.clip(mutant[i, 0], 0, width)
        mutant[i, 1] = np.clip(mutant[i, 1], 0, length)

        
        xi = int(mutant[i, 0])
        yi = int(mutant[i, 1])
        xi_c = xi+1
        yi_c = yi+1
        xi_t = xi-1
        yi_t = yi-1
        xi_c = np.clip(xi_c, 0, VarMaxx)
        yi_c = np.clip(yi_c, 0, VarMaxy)
        xi_t = np.clip(xi_t, 0, VarMaxx)
        yi_t = np.clip(yi_t, 0, VarMaxy)
        if (Area1[yi,xi] == 255
            and Area1[yi,xi_c] == 255 
            and Area1[yi,xi_t] == 255
            and Area1[yi_c,xi] == 255
            and Area1[yi_t,xi] == 255):
            mutant_fitness = fitness_function(mutant)
            if check_connectivity(mutant, communication_range) and mutant_fitness > best_fitness :
                food_sources = np.copy(mutant)
                best_fitness = mutant_fitness
                no_improvement_counters[i] = 0
            else:
                no_improvement_counters[i] += 1

    # Calculate probabilities
    F = np.zeros(num_employed_bees)
    for i in range(1,num_employed_bees):
        property = np.delete(food_sources, i, axis=0)
        F[i] = fitness_function(property)
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

        mutant[selected_food_source, 0] = np.clip(mutant[selected_food_source, 0], 0, width)
        mutant[selected_food_source, 1] = np.clip(mutant[selected_food_source, 1], 0, length)

        
        xi = int(mutant[selected_food_source, 0])
        yi = int(mutant[selected_food_source, 1])
        xi_c = xi+1
        yi_c = yi+1
        xi_t = xi-1
        yi_t = yi-1
        xi_c = np.clip(xi_c, 0, VarMaxx)
        yi_c = np.clip(yi_c, 0, VarMaxy)
        xi_t = np.clip(xi_t, 0, VarMaxx)
        yi_t = np.clip(yi_t, 0, VarMaxy)
        if (Area1[yi,xi] == 255
            and Area1[yi,xi_c] == 255 
            and Area1[yi,xi_t] == 255
            and Area1[yi_c,xi] == 255
            and Area1[yi_t,xi] == 255):
            mutant_fitness = fitness_function(mutant)
            if check_connectivity(mutant, communication_range) and mutant_fitness > best_fitness:
                food_sources[selected_food_source] = mutant[selected_food_source]
                best_fitness = mutant_fitness
                no_improvement_counters[selected_food_source] = 0
            else:
                no_improvement_counters[selected_food_source] += 1

    # Scout bees phase
    for k in range(1, num_employed_bees):
        if no_improvement_counters[k] > limit:
            new_solution=initialize_population()
            mutant_fitness = fitness_function(new_solution)
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

small_radius=1

def draw_circle(ax, center, radius, small_radius):
    # Đường bao  phủ của cảm biến
    outline_circle = plt.Circle(center, radius, fill=False, ec='black', lw=0.8, alpha=1)  
    ax.add_artist(outline_circle)
    
    # Phạm vi cảm biến
    large_circle = plt.Circle(center, radius, color='cyan', alpha=0.2)  
    ax.add_artist(large_circle)

    # Hình tròn nhỏ bên trong
    small_circle = plt.Circle(center, small_radius, fill=False, ec='red', lw=1, alpha=0.7)  
    ax.add_artist(small_circle)
    
def plot_sensor(sensor_nodes, fitness):
    

    fig, (ax,ay) = plt.subplots(1,2)
    ax.set_xlim(0, VarMaxx)
    ax.set_ylim(0, VarMaxy)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, linewidth=0.5)
    ax.invert_yaxis()
    
    ay.set_xlim(0, VarMaxx)
    ay.set_ylim(0, VarMaxy)
    ay.set_aspect('equal', adjustable='box')
    ay.set_xticks(np.arange(0, 101, 10))
    ay.set_yticks(np.arange(0, 101, 10))
    ay.grid(True, linewidth=0.5)
    ay.invert_yaxis()

    # Hình 1: Vẽ các điểm xong và chèn nền

    for i, node in enumerate(sensor_nodes):
        draw_circle(ax,node,Rs,small_radius)
    ax.imshow(image_1, extent=[0, VarMaxx, VarMaxy, 0])
    
    # Hình 2: Vẽ các điểm và Vẽ lại các vật cản

    for i, node in enumerate(sensor_nodes):
        draw_circle(ay,node,Rs,small_radius)
    for bp in ban_position:
        ay.plot(bp[0], bp[1], 'ko')

    plt.grid(True)
    caculator = round((len(ban_position)/((VarMaxy+1)*(VarMaxx+1))),4)
    ax.set_title(f"Coverage percent:{round(fitness/(1-caculator),4) * 100 :.2f} %")
    ay.set_title(f"Coverage percent:{round(fitness/(1-caculator),4) * 100 :.2f} %")
    plt.subplots_adjust(wspace=0.5)
    plt.savefig('graph.pdf')
    plt.show()
with open('ABC.CSV', mode ='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([best_fitness])
    csv_writer.writerows(overall_best_solution)
plot_sensor(overall_best_solution, best_fitness)

