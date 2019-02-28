# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:39:42 2019

@author: jasplund
"""
import mlrose
import numpy as np
import time 
#
## Initialize fitness function object using pre-defined class
#fitness = mlrose.Queens()
#
## Define optimization problem object
#problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize=False, max_val=8)
#
## Define decay schedule
#schedule = mlrose.ExpDecay()
#
## Solve using simulated annealing - attempt 1
#np.random.seed(42)
#               
#init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
#best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10, 
#                                                      max_iters = 1000, init_state = init_state)
#
#print(best_state)
#
#print(best_fitness)
#
#init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
#best_state, best_fitness = mlrose.random_hill_climb(problem, max_attempts=10, 
#                                                    max_iters=1000, restarts=0, init_state=None)
#
#print(best_state)
#
#print(best_fitness)
#
## Solve using simulated annealing - attempt 2
#np.random.seed(42)
#
#best_state, best_fitness = mlrose.genetic_alg(problem,  pop_size=200, mutation_prob=0.1, 
#                                              max_attempts=10, max_iters=1000)
#
#print(best_state)
#
#print(best_fitness)
#
#best_state, best_fitness = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, 
#                                        max_attempts=10, max_iters=1000)
#
#print(best_state)
#
#print(best_fitness)



##################################
##################################
############# 8 Queens ###########
##################################
##################################

## Initialize fitness function object using pre-defined class
#fitness = mlrose.Queens()
#
## Define optimization problem object
#problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize=False, max_val=8)
#
## Define decay schedule
#schedule = mlrose.ExpDecay()
#
## Solve using simulated annealing - attempt 1
#np.random.seed(42)
#               
## Define decay schedule
#schedule = mlrose.ExpDecay()
#
#init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
#best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10, 
#                                                      max_iters = 1000, init_state = init_state)
#
#print(best_state)
#
#
#print(best_fitness)
#
## Solve using simulated annealing - attempt 2
#np.random.seed(42)
#
## Define decay schedule
#schedule = mlrose.ExpDecay()
#
#best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 100, 
#                                                      max_iters = 1000, init_state = init_state)
#
#print(best_state)
#
#print(best_fitness)



# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    
    # Initialize counter
    fitness = 0
    
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):
                
                # If no attacks, then increment counter
                fitness += 1

    return fitness



# Check function is working correctly
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# The fitness of this state should be 22
queens_max(state)



# Initialize custom fitness function object
fitness_cust = mlrose.CustomFitness(queens_max)

schedule = mlrose.ExpDecay()

# Define optimization problem object
problem_cust = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness_cust, maximize = True, max_val = 8)

# Solve using simulated annealing - attempt 1
np.random.seed(42)
bf=0
for i in range(20):
    best_state, best_fitness = mlrose.simulated_annealing(problem_cust, schedule = schedule, 
                                                          max_attempts = 10, max_iters = 1000, 
                                                          init_state = init_state)
    bf += best_fitness
bf/20



print(best_fitness)

## Solve using simulated annealing - attempt 2
#np.random.seed(42)
#
#best_state, best_fitness = mlrose.simulated_annealing(problem_cust, schedule = schedule, 
#                                                      max_attempts = 10, max_iters = 1000, 
#                                                      init_state = init_state)
#
#print(best_state)
#
#print(best_fitness)

init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
bf = 0
for i in range(20):
    best_state, best_fitness = mlrose.random_hill_climb(problem_cust, max_attempts=100, 
                                                        max_iters=1000, restarts=10, init_state=init_state)
    bf += best_fitness
    
bf/20


# Solve using simulated annealing - attempt 2
np.random.seed(42)
bf = 0 
for i in range(20):
    
    best_state, best_fitness = mlrose.genetic_alg(problem_cust,  pop_size=500, mutation_prob=0.2, 
                                                  max_attempts=10, max_iters=1000)
    bf+= best_fitness
    
bf/20

bf = 0
for i in range(20):
        
    best_state, best_fitness = mlrose.mimic(problem_cust, pop_size=500, keep_pct=0.4, 
                                            max_attempts=15, max_iters=1000)
    bf += best_fitness
    
bf/20

#queens_scores.to_csv('./output/queens_score_wine_main.csv')

#############################################
#############################################
######### Travelling salesman ###############
#############################################
#############################################


#Travelling Salesperson Defining Fitness Function as Part of Optimization Problem Definition Step
# Create list of city coordinates
#coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
np.random.seed(42)
n=26
coords_list = [tuple(x) for x in np.random.randint(10, size=(n, 2))]

#def distance(p0,p1):
#    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
#dist_list = []
#for p0, p1 in itertools.combinations(coords_list, 2):
#    dist_list.append((coords_list.index(p0),coords_list.index(p1),-distance(p0, p1)))
#dist_list
# Initialize fitness function object using coords_list
fitness_coords = mlrose.TravellingSales(coords = coords_list)
init_state = np.arange(0,n)
# Define optimization problem object
problem_fit = mlrose.TSPOpt(length = n, fitness_fn = fitness_coords, maximize = True)
schedule = mlrose.ExpDecay()
# Solve using simulated annealing algorithm - attempt 1
np.random.seed(42)

bf = 0
for i in range(20):

    best_state, best_fitness = mlrose.simulated_annealing(problem_fit, schedule = schedule, 
                                                          max_attempts = 1000, max_iters = 1000)
    bf += best_fitness
    
bf/20

# Solve using Randomized Hill Climbing algorithm - attempt 1
np.random.seed(42)
bf = 0 
for i in range(20): 
    
    best_state, best_fitness = mlrose.random_hill_climb(problem_fit, max_attempts=10, 
                                                        max_iters=1000, restarts=10, init_state=init_state)
    bf += best_fitness

bf/20
# Solve using genetic algorithm - attempt 1
np.random.seed(42)

best_state, best_fitness = mlrose.genetic_alg(problem_fit)

print(best_state)

print(best_fitness)

# Solve using genetic algorithm - attempt 2
start = time.time()
np.random.seed(42)
bf = 0 
for i in range(20):
        
    best_state, best_fitness = mlrose.genetic_alg(problem_fit, pop_size = 600, mutation_prob = 0.05, max_attempts = 100)

    bf += best_fitness
    print(best_fitness)
bf/20
end = time.time()
print(end-start)


bf = 0
np.random.seed(42)
start = time.time()
for i in range(20):
    best_state, best_fitness = mlrose.mimic(problem_fit, pop_size=100, keep_pct=0.4, 
                                            max_attempts=5, max_iters=1000)
    bf += best_fitness
bf/20
end = time.time()
print(end-start)



#Travelling Salesperson Using Distance-Defined Fitness Function
# Create list of distances between pairs of cities
#dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), (0, 5, 5.3852), \
#             (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), (1, 3, 2.8284), (1, 4, 2.0000), \
#             (1, 5, 4.1231), (1, 6, 4.2426), (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), \
#             (2, 5, 4.4721), (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
#             (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), (4, 7, 2.2361), \
#             (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]
#
## Initialize fitness function object using dist_list
#fitness_dists = mlrose.TravellingSales(distances = dist_list)
#
#
## Define optimization problem object
#problem_fit2 = mlrose.TSPOpt(length = 8, fitness_fn = fitness_dists, maximize = False)
## Solve using genetic algorithm
#np.random.seed(42)
#
#best_state, best_fitness = mlrose.genetic_alg(problem_fit2, mutation_prob = 0.2, max_attempts = 100)
#
#print(best_state)
#
#print(best_fitness)



#
#best_state, best_fitness = mlrose.genetic_alg(problem_no_fit, mutation_prob = 0.2, max_attempts = 100)
#
#print(best_state)
#
#print(best_fitness)


##################################
##################################
############# k-coloring #########
##################################
##################################


import mlrose
import numpy as np
import networkx as nx

n=50
g = nx.gnp_random_graph(n,0.1)
edges = g.edges()

#edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 3), (3, 4)]
fitness = mlrose.MaxKColor(edges)
init_state = np.random.randint(n, size=n)
#fitness.evaluate(init_state)

schedule = mlrose.GeomDecay()
problem_color = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True, max_val = n)

bf = 0
for i in range(20):
    schedule = mlrose.ExpDecay()
    best_state, best_fitness = mlrose.simulated_annealing(problem_color, schedule = schedule, 
                                                          max_attempts = 1000, max_iters = 1000, 
                                                          init_state = init_state)
    
    bf += best_fitness
    
bf/20

bf = 0
for i in range(20):
    #init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    best_state, best_fitness = mlrose.random_hill_climb(problem_color, max_attempts=1000, 
                                                        max_iters=1000, restarts=0,init_state = init_state)
    bf += best_fitness
bf/20    

# Solve using simulated annealing - attempt 2
np.random.seed(42)
bf = 0
for i in range(20):
    best_state, best_fitness = mlrose.genetic_alg(problem_color,  pop_size=200, mutation_prob=0.001, 
                                                  max_attempts=100, max_iters=1000)
    
    bf += best_fitness
    print(best_fitness)
    
bf/20



bf = 0 
for i in range(10):
    best_state, best_fitness = mlrose.mimic(problem_color, pop_size=200, keep_pct=0.2, 
                                            max_attempts=10, max_iters=1000)
    bf += best_fitness
    
bf/10



