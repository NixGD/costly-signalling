import numpy as np
from population import Population
import random

class Senders(Population):
    def __init__(self, quality, simulation, population_size = None):
        self.quality = quality  # quality = acceptance_reward / cost.  Perhaps use the inverse
        Population.__init__(self, simulation, population_size)

    def calculate_payoff(self, sender_num):
        """
        :param Int signal: The signal/strategy that one individual plays.
        :return:  A normalized value between -1 and 1.
        """
        signal = self.strategies[sender_num]
        acceptance = self.simulation.get_avg_acceptance(self.quality, sender_num)

        # cost = signal * full_cost = signal * acceptance_reward / quality = signal / quality
        payoff = acceptance - signal / self.quality
        return payoff

    @staticmethod
    def get_random_strategy():
        """
        Returns a random value in the sender range
        """
        return random.uniform(0,1)

    def vary_strategy(self, strategy):
        sigma = self.simulation.sender_sigma
        return np.clip(sigma * np.random.randn() + strategy, 0, 1)

    def strategy_points(self):
        return [strategy for s_list in self.strategy_history for strategy in s_list]

    def high_means(self):
        return [np.mean(s_list) for s_list in self.strategies]

    def sender_heatmap(self, gBins, iBins):
        generation_points = [g for g in range(self.simulation.num_generations)  \
                               for _ in range(self.population_size)]   #[0,0,...,0,1,1,...1,2,2...]
        extent =[[0, self.simulation.num_generations], [0, 1]]
        heatmap, x_edges, y_edges = np.histogram2d(generation_points, self.strategy_points(), bins=(gBins, iBins), range = extent)
        max_density = sum([np.max(generation) for generation in heatmap]) / gBins
        adjusted_heatmap = heatmap/max_density
        return (adjusted_heatmap, x_edges, y_edges)
