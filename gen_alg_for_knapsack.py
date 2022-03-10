from random import choices, randrange, sample
from typing import Callable, List, Tuple
from collections import namedtuple
from random import randint, random
from functools import partial
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
import concurrent.futures
from numba import jit, cuda


# %matplotlib inline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# data=pd.read_csv('cleaned_players_19_20.csv',encoding = "ISO-8859-1", low_memory=False,sep='\t')
# data=pd.read_csv('cleaned_players_20_21.csv')
data=pd.read_csv('cleaned_players_21_22.csv')
# data.head(667)

data=data.drop(data.loc[:, 'goals_scored':'assists'].columns, axis = 1)

data=data.drop(data.loc[:, 'minutes':'selected_by_percent'].columns, axis = 1)

data=data.drop(data.loc[:, 'team_name':'team_rank'].columns, axis = 1)

data['full_name'] = data[['first_name','second_name']].apply(lambda x: ' '.join(x), axis=1)

data=data.drop(data.loc[:, 'first_name':'second_name'].columns, axis = 1)

above_120 = data[data["total_points"] > 80]
above_120.rename(columns={'total_points': 'value', 'now_cost': 'weight','full_name':'name'}, inplace=True)

first_column = above_120.pop('name')
  
# insert column using insert(position,column_name,first_column) function
above_120.insert(0, 'name', first_column)
above_120=above_120.sort_values('Position')
# above_120=above_120.sort_values(["Position", "value"], ascending = (True, True))
available_gkp_no=len(above_120[above_120['Position'] == 0])
available_def_no=len(above_120[above_120['Position'] == 1])
available_mid_no=len(above_120[above_120['Position'] == 2])
available_fwd_no=len(above_120[above_120['Position'] == 3])

formation_gkp_no = 1    
formation_def_no = 4
formation_mid_no = 5
formation_fwd_no = 1
'''
Genome is the genetic representation of our solution
A list of 0s and 1s where 1s indicate player selected, and
0s indicate otherwise
'''
Genome = List[int]

'''
A population is simply a list of genomes
'''
Population = List[Genome]

'''
FitnessFunc Takes a genome and returns a fitness value to make a 
correct choice
'''
FitnessFunc=Callable[[Genome],int]

'''
Populate function takes nothing but spits out new solutions
'''
PopulateFunc = Callable[[],Population]

'''
Selection function takes a population and a fitness function
to select two solutions to be the parents of our next generation's
solutions
'''
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]

'''
A crossover function takes two genomes and generates two new
genomes
'''
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]

'''
A mutation function also takes one genome and sometimes returns a modified one
based on probabilities
'''
MutationFunc = Callable[[Genome], Genome]

Thing = namedtuple('Thing',['name','value','weight'])

things = list(above_120.itertuples(name='Thing', index=False))

'''
Things is a structure with a name, points and cost
'''
# things=[
#     Thing('Laptop',500,2200),
#     Thing('Headphones',150,160),
#     Thing('Mug',60,350),
#     Thing('Notepad',40,333),
#     Thing('Bottle',30,192),
# ]
# things=[
#     Thing('A',71,29),
#     Thing('B',155,75),
#     Thing('C',217,118),
#     Thing('D',324,215),
#     Thing('E',431,311),
#     Thing('F',493,330),
#     Thing('G',499,334),
#     Thing('H',543,368),
#     Thing('I',609,431),
#     Thing('J',752,536),
#     Thing('K',936,697),
#     Thing('L',1189,935),
#     Thing('M',1349,1059),
#     Thing('N',1479,1170),
#     Thing('O',1693,1366),
# ]
# things=[
#     Thing('player1','50','130'),
#     Thing('player2','60','150'),
#     Thing('player3','55','125'),
#     Thing('player4','90','180'),
#     Thing('player5','110','210'),
# ]
#2:39 for more_things

'''We generate a random list of 1s and 0s of length of list
 of players we want to choose from'''
def generate_genome(length: int) -> Genome:
    # genome_zeroes = [0] * length
    # rand_list=sample(range(0, length), 11)
    # for item in rand_list:
    #     genome_zeroes[item]=1
    genome_gkp = [0] * available_gkp_no
    genome_def = [0] * available_def_no
    genome_mid = [0] * available_mid_no
    genome_fwd = [0] * available_fwd_no
    rand_list=sample(range(0, available_gkp_no), formation_gkp_no)
    for item in rand_list:
        genome_gkp[item]=1
    rand_list=sample(range(0, available_def_no), formation_def_no)
    for item in rand_list:
        genome_def[item]=1
    rand_list=sample(range(0, available_mid_no), formation_mid_no)
    for item in rand_list:
        genome_mid[item]=1
    rand_list=sample(range(0, available_fwd_no), formation_fwd_no)
    for item in rand_list:
        genome_fwd[item]=1
    genome=[]
    genome.extend(genome_gkp)
    genome.extend(genome_def)
    genome.extend(genome_mid)
    genome.extend(genome_fwd)
    return genome
    # return choices([0,1], k=length)

'''
To generate a population, we call our genome generation function 
multiple times until our population size meets
'''
def generate_population(size: int, genome_length: int) -> Population:
    # start_popn_gen = time.time()
    # popn=[]
    # for _ in range(size):
    #     popn.append(generate_genome(genome_length))
    # end_popn_gen = time.time()
    # print(f"popn gen time:{end_popn_gen - start_popn_gen}s")
    # return popn
    return [generate_genome(genome_length) for _ in range(size)]

# def generate_population(size: int, genome_length: int) -> Population:
#     start_popn_gen = time.time()
#     popn=[]
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = [executor.submit(generate_genome, genome_length) for _ in range(size)]
#     for f in concurrent.futures.as_completed(results):
#         # print(f.result)
#         popn.append(f.result())
#     end_popn_gen = time.time()
#     print(f"popn gen time:{end_popn_gen - start_popn_gen}s")
#     return popn
'''
Fitness function to evaluate our genomes
it takes genomes, the list of things we can choose from,
and a weight_limit(cost limit) parameter
returns the fitness value, if not fit according to the conditions
returns 0
'''
def fitness(genome: Genome, things: [Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and things must be of same length")
    
    weight = 0
    value = 0

    if genome.count(1) != 11:
        return 0
    
    genome1=genome[0:available_gkp_no]
    if genome1.count(1) != formation_gkp_no:
        return 0

    genome2=genome[available_gkp_no:(available_gkp_no+available_def_no)]
    if genome2.count(1) != formation_def_no:
        return 0

    genome3=genome[(available_gkp_no+available_def_no):(available_gkp_no+available_def_no+available_mid_no)]
    if genome3.count(1) != formation_mid_no:
        return 0
    
    genome4=genome[(available_gkp_no+available_def_no+available_mid_no):(available_gkp_no+available_def_no+available_mid_no+available_fwd_no)]
    if genome4.count(1) != formation_fwd_no:
        return 0

    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value

            if weight > weight_limit:
                return 0

    teams = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            teams.append(thing.team_id)
    
    for items in teams:
        if teams.count(items) > 3:
            return 0

    return value

'''
selection function selects a pair of solutions which will be
the parents of two new solutions of next generation
solutions of higher fitness should be more likely to be chosen
'''
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    # start_fitness_calc = time.time()
    # weights=[]
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = [executor.submit(fitness_func, genome, things, 850) for genome in population]
    #     print(results)
    # for f in concurrent.futures.as_completed(results):
    #     # print("OK")
    #     weights.append(f.result())
    # # n = 0
    # # for genome in population:
    # #     weights.append(fitness_func(genome))
    # #     n += 1
    # end_fitness_calc = time.time()
    # print(f"Fitness calculation time for popn this gen:{end_fitness_calc - start_fitness_calc}s")
    # # print(n)

    return choices(
        population = population,
        weights=[fitness_func(genome) for genome in population],
        # weights = weights,
        k=2
        #k=2 means we draw twice from our population to get a pair
    )
    
'''
takes two genomes as parameters and returns two genomes as output
'''
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    #genomes should be of same length
    if len(a) != len(b):
        raise ValueError("genomes a and b must be of same length")
    
    #lengths of genome should be atleast 2 otherwise crossover mkes no sense
    length = len(a)
    if length < 2:
        return a, b
    
    '''
    we randomly choose an index to cut the genomes in half
    and take the first half of genome a and second part of genome b 
    put them together and return them as our first solution
    for second solution we take first part of genome b and second part of
    genome a and put them together to return it as our second solution
    '''
    p = randint(1, length-1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

    # p1 = randint(1, available_gkp_no-1)
    # child1a = a[0:p1] + b[p1:available_gkp_no]
    # child1b = b[0:p1] + a[p1:available_gkp_no]

    # p1 = randint(1, available_def_no-1)
    # child2a = a[available_gkp_no:available_gkp_no+p1] + b[available_gkp_no+p1:available_gkp_no+available_def_no]
    # child2b = b[available_gkp_no:available_gkp_no+p1] + a[available_gkp_no+p1:available_gkp_no+available_def_no]

    # p1 = randint(1, available_mid_no-1)
    # child3a = a[available_gkp_no+available_def_no:available_gkp_no+available_def_no+p1] + b[available_gkp_no+available_def_no+p1:available_gkp_no+available_def_no+available_mid_no]
    # child3b = b[available_gkp_no+available_def_no:available_gkp_no+available_def_no+p1] + a[available_gkp_no+available_def_no+p1:available_gkp_no+available_def_no+available_mid_no]

    # p1 = randint(1, available_fwd_no-1)
    # child4a = a[available_gkp_no+available_def_no+available_mid_no:available_gkp_no+available_def_no+available_mid_no+p1] + b[available_gkp_no+available_def_no+available_mid_no+p1:available_gkp_no+available_def_no+available_mid_no+available_fwd_no]
    # child4b = b[available_gkp_no+available_def_no+available_mid_no:available_gkp_no+available_def_no+available_mid_no+p1] + a[available_gkp_no+available_def_no+available_mid_no+p1:available_gkp_no+available_def_no+available_mid_no+available_fwd_no]

    # child_a = child1a + child2a + child3a + child4a
    # child_b= child1b + child2b + child3b + child4b

    # return child_a, child_b

'''
Mutation function takes genome and with certain probability changes
0s to 1s and 1s to 0s at random positions

'''
def mutation(genome: Genome, num: int =1, probability: float = 0.25) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        '''
        for mutation we use random index and if random returns a value higher than
        probability we leave it alone, otherwise it is in our mutation probability
        and we need to change it to be the absolute value of the current value-1
        eg: if genome[index]=1 then abs(1-1)=0
        if genome[index]=0 then abs(0-1)=1
        '''
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    
    return genome

def genome_fitness(genome: Genome, things: [Thing]) -> int:
    fitness_score = 0
    for i, thing in enumerate(things):
        if genome[i] == 1:
            fitness_score += thing.value
    
    return fitness_score

def genome_weight(genome: Genome, things: [Thing]) -> int:
    total_weight = 0
    for i, thing in enumerate(things):
        if genome[i] == 1:
            total_weight += thing.weight
    
    return total_weight

'''
This function actually runs the evolution

'''
def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int, # if fitness of the best solution exceeds this limit we've reached our goal
    selection_func: SelectionFunc = selection_pair,#these three functions are initialized with our
    crossover_func: CrossoverFunc =single_point_crossover,#default implementations
    mutation_func: MutationFunc = mutation,
    generation_limit: int =100 #max no. of generations our evolution runs for if it does not reach the fitness limit before that
) -> Tuple[Population, int]:
    '''
    getting first ever generation by calling the populate fn.
    '''
    population = populate_func()

    '''
    looping for genertion limit 
    '''
    for i in range(generation_limit):
        '''
        Sorting our population by fitness
        this way we know our top solution are inheriting the first indices
        of our list of genomes, 
        '''
        population = sorted(
            population,
            key = lambda genome: fitness_func(genome),
            reverse = True
        )

        '''
        Sorting comes in handy when we want to check 
        if we have already reached the fitness limit and return early from our loop
        
        '''
        no_of_fit_genomes_in_this_generation = 0
        for k in range(len(population)):
            if fitness_func(population[k]) != 0:
                no_of_fit_genomes_in_this_generation +=1
        
        print("No. of unique genomes this generation:",len(set(map(tuple, population))))
        print("No. of non zero fitness genomes in this generation is:",no_of_fit_genomes_in_this_generation)
        print(f"Max fitness from population of this generation: {genome_fitness(population[0], things)}")
        print(f"Weight of the fittest genome of this generation: {genome_weight(population[0], things)}")

        if fitness_func(population[0]) >= fitness_limit:
            break
        
        '''
        or if we want to implement elitism and just keep our top two solutions for
        our next generation
        '''
        next_generation = population[0:2]

        '''
        Generating all other new solutions for our next generation
        we pick two parents and get two new solutions everytime
        so we loop for half the length of a generation to get as many solutions in our next generation as before
        '''
        for j in range(int(len(population) / 2) -1): #looping one less time because we've already copied the top top solutions from our last generation
            #in each loop we call selection function to get our parents
            parents = selection_func(population, fitness_func)
            #we put these selected parents into crossover function to get two child solutions for our next generation
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            #applying mutation function for each offspring
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        #replacing current population with the next generation
        population = next_generation
        print("Generation:",i+1,"Completed")

    #sorting the population to check if it reached fitness limit in this generation
    population =sorted( 
        population,
        key = lambda genome: fitness_func(genome),
        reverse=True
    )

    #returning the population and no. of generations we are in
    
    return population, i

start = time.time()
population, generations =run_evolution(
    populate_func = partial(
        generate_population, size = 500, genome_length = len(things)
    ),
    fitness_func = partial(
        fitness, things= things, weight_limit = 850
    ),
    # fitness_limit = 2131,#for 19/20 season
    fitness_limit = 2218, #for 18/19 season
    generation_limit = 100
)
end = time.time()
def genome_to_things(genome: Genome, things: [Thing]) -> [Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing.name]

    return result

print(f"number of generations: {generations + 1}")
print(f"time:{end - start}s")
print(f"best solution: {genome_to_things(population[0], things)}")
print(f"best solution fitness: {genome_fitness(population[0], things)}")
print(f"best solution total weight: {genome_weight(population[0], things)}")


# def weight_func(genome: Genome, things: [Thing]) -> int:
#     weight = 0
#     value = 0
#     for i, thing in enumerate(things):
#         if genome[i] == 1:
#             weight += thing.weight
#             value += thing.value
    
#     return weight



