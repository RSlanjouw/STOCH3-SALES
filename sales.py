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


def read_file(file="eil51.tsp.txt"):
    folder= "TSP-Configurations/"
    file = open(folder + file, "r")
    lines = file.readlines()
    file.close()

    # skip the bs!
    lines = lines[6:-1]
    lines = [line.split() for line in lines]
    lines = [[int(line[0]), float(line[1]), float(line[2])] for line in lines]
    return lines

def random_replacement(route):
    new_route = route.copy()  # Zorg dat de route echt gekopieerd wordt
    i, j = np.random.choice(range(1, len(route) - 1), 2, replace=False)  # Alleen interne steden wisselen
    temp1 = new_route[j].copy()
    temp2 = new_route[i].copy()
    new_route[i] = temp1
    new_route[j] = temp2
    return new_route

def random_route(coordinates):  
    start = coordinates[0]
    end = coordinates[0]
    cities = coordinates[1:]
    route = [start] + list(np.random.permutation(cities))
    return np.array(route)


def greedy_algorithm(coordinates):
    start = coordinates[0]
    # choose closest not yet visited city
    cities = coordinates[1:]
    route = [start]
    current_city = start
    while len(cities) > 0:
        distances = [distance_two_nodes(current_city, city) for city in cities]
        closest_city = cities[np.argmin(distances)]
        route.append(closest_city)
        cities.remove(closest_city)
        current_city = closest_city

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


def two_opt_improvement(route):
    new_route = route.copy()
    for i in range(500000):
        new_route = two_opt_algorithm(new_route)
        if distance_route(new_route) < distance_route(route):
            route = new_route
        if i % 10000 == 0:
            print(i)
    return route

def random_improvement(route):
    new_route = route.copy()
    for i in range(10000):
        new_route = random_replacement(new_route)
        if distance_route(new_route) < distance_route(route):
            route = new_route
    return route

coordinates = read_file()
coordinates = np.array(coordinates)
route = random_route(coordinates)
# visualize the route in the grid
plt.plot(route[:,1], route[:,2], 'o-')
plt.show()
print(distance_route(route))

route = two_opt_improvement(route)
print(distance_route(route))
plt.plot(route[:,1], route[:,2], 'o-')
plt.show()
print(route)



def get_optimum(route):
    # read eil51.opt.tour.txt
    folder= "TSP-Configurations/"
    file = open(folder + "eil51.opt.tour.txt", "r")
    lines = file.readlines()
    file.close()
    lines = lines[5:-2]
    lines = [int(line) for line in lines]

    empty_list = [0] * len(lines)
    for item in route:
        city_number = item[0]
        print(city_number)
        matching_index = lines.index(int(city_number))
        print(matching_index)
        empty_list[matching_index] = item
        print(empty_list)
    return empty_list

# route_best = get_optimum(route)
# print(distance_route(route_best))