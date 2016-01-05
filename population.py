import math
import random

class Population:
    def __init__(self, simulation, population_size):
        self.simulation = simulation
        self.strategy_history = [None]*simulation.num_generations
        self.population_size = population_size
        self.strategies = [self.get_random_strategy() for _ in range(self.population_size)]
        self.strategy_history[0] = self.strategies

    def update(self):
        fitness_list = [self.calculate_fitness(individual) for individual in range(self.population_size)]
        self.strategies = [self.get_new_strategy(self.strategies, fitness_list) for _ in range(self.population_size)]
        self.strategy_history[self.simulation.i] = self.strategies

    def calculate_fitness(self, strategy):
        """
        Takes in a single strategy, returns the fitness.
        """
        payoff = self.calculate_payoff(strategy)
        return math.exp(self.simulation.selection_strength * payoff)

    def get_new_strategy(self, old_strategies, fitness_list):
        """
        Takes in the strategy list, the fitness list and the mutation rate.
        Each child then either mutates randomly or takes on the strategy of a parent.
        Each parent is chosen with chance proportional to their fitness.
        The function returns the strategies of the children.
        """
        if random.random() < self.simulation.mutation_rate:
            new_strategy = self.get_random_strategy() # Takes random strategy if mutates
        else:
            parent_strategy = old_strategies[self.find_parent(fitness_list)]  # takes parent strategy otherwise
            new_strategy = self.vary_strategy(parent_strategy)  # with some small variation.
        return new_strategy

    @staticmethod
    def find_parent(fitness_list):
        fraction_sum = random.random() * sum(fitness_list)
        parent = -1
        fitness_sum = 0
        while fitness_sum < fraction_sum:  # Walk through the list until the sum is too high, when we'll have the parent
            parent += 1
            fitness_sum += fitness_list[parent]
        return parent
