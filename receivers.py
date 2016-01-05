from population import Population
import random
import numpy as np
import math
import itertools as it

def pair_avg(iterable):
    "s -> avg(s0,s1), avg(s1,s2), avg(s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return map(lambda a,b: (a+b)/2, a, b)


def pair_list(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a,b)


class Receivers(Population):
    def __init__(self, simulation, population_size = None):
        Population.__init__(self, simulation, population_size)

    @staticmethod
    def get_random_strategy():
        """
        Returns a list of two numbers representing the range that the receiver will accept.
            Every value in [0, 1) has equal chance of being included in range.
        """
        width = random.random()*0.5
        center = random.uniform(-width, 1+width)
        low = max(0, center-width)
        high = min(1, center+width)
        return [low, high]

    def calculate_payoff(self, receiver_num):
        accepted = self.simulation.get_aceptees(receiver_num)
        payoff = sum([math.log(q, 2)*num_accepted for q, num_accepted in accepted.items()])
        return payoff

    def vary_strategy(self, strategy):
        sigma = self.simulation.receiver_sigma
        strategy[0] = np.clip(sigma * np.random.randn() + strategy[0], 0, 1)
        strategy[1] = np.clip(sigma * np.random.randn() + strategy[1], 0, 1)
        return strategy

    def avg_acceptance_level(self, gen_tuple, signal):
        gen_list = range(int(gen_tuple[0]), int(gen_tuple[1]))
        acceptances_list = [sum([self.get_acceptance_individual(signal, r_strategy) for r_strategy in self.strategies[generation]])
            for generation in gen_list]
        avg_acceptances = sum(acceptances_list)/len(acceptances_list)
        return avg_acceptances/self.population_size

    def get_acceptance_profile(self, gen_edges, signal_edges):
        """
        :param gen_edges:
        :param signal_edges:
        :return:   A numpy array of generation profiles.
        """

        return np.array([[self.avg_acceptance_level(generation_tuple, signal)
                            for signal in pair_avg(signal_edges) ]
                            for generation_tuple in pair_list(gen_edges)])

    def get_acceptance_population(self, sender_strategy):
        return sum([self.get_acceptance_individual(sender_strategy, strategy) for strategy in self.strategies])

    @staticmethod
    def get_acceptance_individual(sender_strategy, receiver_strategy):
        if receiver_strategy[0] < sender_strategy < receiver_strategy[1]:
            return 1
        return 0