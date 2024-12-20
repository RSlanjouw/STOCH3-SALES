# read eil51.tsp.txt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def distance_two_nodes(node1, node2):
    x1, y1 = node1[1], node1[2]
    x2, y2 = node2[1], node2[2]
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def distance_route(route):
    total_distance = 0
    for i in range(len(route)-1):
        total_distance += distance_two_nodes(route[i], route[i+1])
    total_distance += distance_two_nodes(route[-1], route[0])
    return total_distance


def read_file(file="a280.tsp.txt"):
    folder= "TSP-Configurations/"
    file = open(folder + file, "r")
    lines = file.readlines()
    file.close()

    # skip the bs!
    lines = lines[6:-1]
    lines = [line.split() for line in lines]
    lines = [[int(line[0]), float(line[1]), float(line[2])] for line in lines]
    return lines


def random_route(coordinates):  
    start = coordinates[0]
    end = coordinates[0]
    cities = coordinates[1:]
    route = [start] + list(np.random.permutation(cities))
    return np.array(route)


def two_opt_algorithm(route):
    new_route = route.copy() 

    indices = np.arange(len(route))
    i = np.random.choice(indices[1:-1])
    j = i+1
    # remove from indices
    if i != 1 and i != len(route) - 2:
        indices = np.delete(indices, [i-1,i,i+1, i+2])
    elif i == 1:
        indices = np.delete(indices, [i,i+1, i+2])
    else:
        indices = np.delete(indices, [i-1,i,i+1])
    p = np.random.choice(indices[1:-1])
    q = p + 1
    
    tempj = new_route[j].copy()
    tempp = new_route[p].copy()

    # write i,p and j,q in the new route
    new_route[j] = tempp
    new_route[p] = tempj
    # reverse the order of the cities between p and j or j and p
    if p < j:
        new_route[p+1:j] = route[p+1:j][::-1]
    else:
        new_route[j+1:p] = route[j+1:p][::-1]
    
    # new_route += [new_route[0]]
    return new_route
    
def simulated_annealing(route, initial_temp=1000, cooling_rate=0.995, iterations=10000):

    current_route = route
    current_distance = distance_route(route)
    best_route = current_route
    best_distance = current_distance
    temperature = initial_temp
    final_temp = 1
    results = []

    for iteration in range(iterations):
        # Generate a neighboring solution using 2-opt
        new_route = two_opt_algorithm(current_route)
        new_distance = distance_route(new_route)

        if new_distance < current_distance or np.random.rand() < np.exp((current_distance - new_distance) / temperature):
            current_route = new_route
            current_distance = new_distance

            if current_distance < best_distance:
                best_route = current_route
                best_distance = current_distance

        if coolingsc == "lin":
            temperature = max(final_temp, temperature - cooling_rate * iteration)
        if coolingsc == "exp":
            temperature *= cooling_rate
        if coolingsc == "log":
            temperature = temperature / (1 + np.log(1 + iteration))

        
        

        if iteration % 1000 == 0 or iteration == iterations - 1:
            print(f"Iteration {iteration}, Best Distance: {best_distance:.6f}, Temperature: {temperature:.6f}")
            #plt.plot(best_route[:, 1], best_route[:, 2], 'o-', label='Best Route')
            #plt.title(f"Iteration {iteration}")
            #plt.pause(0.01)
    plt.show()
    return best_route, best_distance

# Main execution
coordinates = read_file()
initial_route = random_route(coordinates)


initial_route = random_route(coordinates)

for x in range(100):
    best_route, best_distance = simulated_annealing(initial_route)