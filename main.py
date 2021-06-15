import json
import client
import numpy as np
import population as pp

KEY = 'ul9p5fKHuHUv3na92PQQ0d3gvvfcWayvCeODJ2lbbh5WuF4fTQ'

GENE_MIN = -10
GENE_MAX = 10
POPULATION_SIZE = 11
PARENT_PASSED = 5
CHROMOSOME_SIZE = 11
MUTATE_PROB = 4/11
MATING_POOL_SIZE = 6
CROSSOVER = 3
NUMBER_OF_GENERATION = 5

X = 1

Y = 1

population = []

with open('overfit.txt','r') as readfile:
    data = json.load(readfile)


def create_mating_pool(population_fitness : list):
    mating_pool = population_fitness[:MATING_POOL_SIZE]
    return mating_pool

def mutate(mutate_vec):
    
    for i in range(len(mutate_vec)):
        for j in range(11):
            if np.random.uniform(0,1) < MUTATE_PROB:
                mutate_vec[i][j] = mutate_vec[i][j]*np.random.uniform(0.9,1.1)
                if np.random.uniform(-1,+1) < 0:
                    mutate_vec[i][j] *= -1
                if mutate_vec[i][j] > 10:
                    mutate_vec[i][j] = 10
                elif mutate_vec[i][j] < -10:
                    mutate_vec[i][j] = -10 
    return mutate_vec

def generate_initial_population(overfit_arr):
    if len(overfit_arr) != CHROMOSOME_SIZE:
        raise ValueError
    temp = [list(overfit_arr) for i in range(POPULATION_SIZE)]
    temp = mutate(temp)
    temp[0] = overfit_arr
    return temp

def errrorToFitness(train_err,validation_err):
    return -(abs(train_err - validation_err))
    # return -(X*train_err + Y*validation_err)  # X = 0.3 AND Y = 0.7

def crossover( mom : list, dad : list):
    thresh = np.random.randint(CHROMOSOME_SIZE)
    alice = mom.copy()
    bob = dad.copy()
    alice[0:thresh] = mom[0:thresh]
    bob[0:thresh] = dad[0:thresh]
    return alice,bob


def normal_crossover(mom : list, dad : list):
    thresh = np.random.randint(CHROMOSOME_SIZE)
    child = dad.copy()
    child[0:thresh] = mom[0:thresh]
    return child

def get_fitness(population_p):
    
    weight_fitness_err = []
    fitness = []
    train_error = []
    validation_error = []

    for current_pop in population_p:
        train_err , valid_err = client.get_errors(KEY,(current_pop)) 
        fit_curr = errrorToFitness(train_err,valid_err)
        weight_fitness_err.append((current_pop,train_err,valid_err,fit_curr))    
        fitness.append(fit_curr)
        train_error.append(train_err)
        validation_error.append(valid_err)
    
    return weight_fitness_err,  train_error, validation_error, fitness

def normal_breeding(population_data):

    weight_fitness_err,  train_error, validation_error, fitness = get_fitness(population_data)

    weight_fitness_err.sort(key=lambda x: x[3],reverse=True)

    mating_pool = create_mating_pool(weight_fitness_err)
    
    offspring = []

    for i in range(POPULATION_SIZE - PARENT_PASSED):
        rm1 = np.random.random_integers(0,MATING_POOL_SIZE - 1)
        mom = mating_pool[rm1][0]
        rm2 = np.random.random_integers(0,MATING_POOL_SIZE - 1)
        dad = mating_pool[rm2][0]
        if rm1 == rm2:
            child = normal_crossover(mom,dad)
            offspring.append(child)
        else:
            alice,bob = crossover(mom,dad)
            offspring.append(alice)
            offspring.append(bob)

    for i in range(PARENT_PASSED):
        offspring.append(mating_pool[i][0])      

    mutate(offspring)
    
    return offspring,weight_fitness_err 


populationCompletedata = generate_initial_population(data)

weight_fitness = []
minimum_validation = []
minimum_train = []
maximum_fit = []

# outer loop for algo 
miniFit = []
for i in range(NUMBER_OF_GENERATION):

    temp = np.array(populationCompletedata,copy=True)
    temp1 = np.mean(temp,axis = 0)

    populationCompletedata,weight_fitness  =  normal_breeding(populationCompletedata)

final_array = []
for i in range(10):
    final_array.append(weight_fitness[i][0])

with open('output.txt','a') as writef:
    json.dump(final_array, writef)
