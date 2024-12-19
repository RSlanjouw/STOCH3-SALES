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
    folder= "Dropbox/Rinske/Computational Science/Stochastic Simulation/TSP/"
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
    route = [start] + list(np.random.permutation(cities)) + [end]
    return np.array(route)



def two_opt_alg(route):
    new_route = route.copy()
    n= len(route)
    # Kies i willekeurig uit de mogelijke waarden: 1 tot n-3
    i = np.random.choice(np.arange(1, n-2))

    # j is altijd i + 1
    j = i + 1

    # p is altijd minstens j+1 en maximaal n-1
    p = np.random.choice(np.arange(j+1, n))

    # q is altijd p + 1
    q = p + 1

    # to_swap = new_route[j:p].copy()
    new_route[j:p] = new_route[j:p][::-1]
    # Verwissel de volgorde van de steden

    return new_route
    
def simulated_annealing(route, initial_temp=10000, cooling_rate=0.995, iterations=100000, coolingsc = "exp", markov_chain_length=1, verbose=True):

    current_route = route
    current_distance = distance_route(route)
    best_route = current_route
    best_distance = current_distance
    temperature = initial_temp
    minimum = False
    final_temp = 1
    results = []
    step_size= initial_temp/ (cooling_rate * iterations)
    temp = []

    for iteration in range(iterations):
        # Generate a neighboring solution using 2-opt
        new_route = two_opt_alg(current_route)
        new_distance = distance_route(new_route)

        if new_distance < current_distance or np.random.rand() < np.exp((current_distance - new_distance) / temperature):
            current_route = new_route
            current_distance = new_distance

            if current_distance < best_distance:
                best_route = current_route
                best_distance = current_distance

        if iteration % markov_chain_length == 0:
            if coolingsc == "lin":
                temperature = max(10**-7,temperature - step_size)
            if coolingsc == "exp":
                temperature = max(10**-7,temperature * cooling_rate)
            if coolingsc == "log":
                temperature = max(10**-7, temperature / (1 + np.log(1 + iteration)))

        if iteration % 1000 == 0 or iteration == iterations - 1:
            results.append(best_distance)
            temp.append(temperature)
        if verbose and (iteration % 10000 == 0 and iteration != 0):
            print(f"Iteration: {iteration}, Best distance: {best_distance}, Temperature: {temperature}")
    return best_route, best_distance, results, temp

# Main execution
coordinates = read_file()
initial_route = random_route(coordinates)

markov_lengths = [1,10,50,100]
results = []

# best_route, best_distance, allresults1, temp = simulated_annealing(initial_route, markov_chain_length = 1)
# best_route, best_distance, allresults2, temp = simulated_annealing(initial_route, iterations=1000000 ,markov_chain_length = 500)

# xs = np.linspace(0,100000, 101)

# plt.plot(xs, allresults1, label = 1)
# plt.plot(xs,allresults2, label = 500)
# plt.legend()
# plt.show()


for markov in markov_lengths:
    best_distances = []
    distances = []
    temps = []
    for i in range(5):
        initial_route = random_route(coordinates)
        best_route, best_distance, allresults, temp = simulated_annealing(initial_route, markov_chain_length = markov, iterations=300000)
        best_distances.append(best_distance)
        distances.append(allresults)
        temps.append(temp)
    distances_array = np.array(distances) 
    results.append([ markov, np.mean(best_distance), np.array(np.mean(distances_array, axis=0)), np.mean(np.array(temp))])
    print(f"Done with markov length {markov}")

for res in results:
   plt.plot(range(1, len(res[2]) + 1),res[2], label = rf"ML={res[0]}, $\xi$={res[1]:.2f}")
plt.axhline(y=2586.77, color = 'black',linestyle='dashed', label = "optimal route = 2586.77")
plt.xlabel("Iterations (x1000)")
plt.ylabel("Mean Route Distance")
plt.grid(True)
plt.legend()
plt.savefig("Markov_1_10_50_100.png")
plt.show()


# # write into csv
# df = pd.DataFrame(results, columns=["Markov Chain Lenghth","Mean Best Distance", "Mean Distances", "Mean temp"])
# df.to_csv("exp_5times_markov.csv")


