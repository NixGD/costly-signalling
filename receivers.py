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

    def calculate_payoff(self, receiver_num):
        accepted = self.simulation.get_aceptees(receiver_num)
        payoff = sum([math.log(q, 2)*num_accepted for q, num_accepted in accepted.items()])
        return payoff

    def get_acceptance_profile(self, gen_edges, signal_edges):
        """
        :param gen_edges:
        :param signal_edges:
        :return:   A numpy array of generation profiles.
        """
        return np.array([[self.avg_acceptance_level(generation_tuple, signal)
                            for signal in pair_avg(signal_edges) ]
                            for generation_tuple in pair_list(gen_edges)])

    def avg_acceptance_level(self, gen_tuple, sender_strategy):
        """
        Helper for acceptance profile.
        :param gen_tuple: The range of generations that this pixel covers.
        :param sender_strategy: The strategy being tested
        :return: The acceptance of this strategy of an average receiver over this time period
        """
        gen_list = range(int(gen_tuple[0]), int(gen_tuple[1]))
        return np.mean([self.get_acceptance_population(sender_strategy, self.strategies[gen]) for gen in gen_list])

    def get_acceptance_population(self, sender_strategy, gen):
        """
        Helper for acceptance profiling
        :param sender_strategy: The strategy being tested
        :param gen: in specified generation
        :return: The average acceptance of receivers in that generation of said strategy.
        """
        receiver_strategies = self.strategy_history[gen]
        return np.mean([self.get_acceptance_individual(sender_strategy, strategy) for strategy in receiver_strategies])


class HighLow(Receivers):
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
        return low, high

    def vary_strategy(self, strat):
        sigma = self.simulation.receiver_sigma
        low  = np.clip(sigma * np.random.randn() + strat[0], 0, 1)
        high = np.clip(sigma * np.random.randn() + strat[1], 0, 1)
        return low, high

    @staticmethod
    def get_acceptance_individual(sender_strategy, receiver_strategy):
        return receiver_strategy[0] < sender_strategy < receiver_strategy[1]
