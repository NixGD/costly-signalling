import numpy as np
from senders import Senders
from receivers import Receivers, HighLow
import pickle
import matplotlib.pyplot as plt


def rgb_map(density_list):
    [low, high, receivers] = density_list
    rgb = [1-low, 1-high-low, 1-high]
    rgb_clipped = [np.clip(x, 0, 1) for x in rgb]
    alpha = np.clip(high+low, 0, 1)
    darkness = .4
    rgb_receivers = [overlay_grey(fg, alpha, (1 - receivers*darkness)) for fg in rgb_clipped]
    return rgb_receivers


def overlay_grey(fg, fg_alpha, bg):
    # All inputs scaled 0-1. Fg (foreground) with alpha + greyscale background.
    return ((1 - fg_alpha) * bg) + (fg_alpha * fg)


class Simulation:
    def __init__(self, num_generations = 50000, receiver_type = HighLow,
                 senders = ((.5, 50), (2, 25)), receiver_number = 50,
                 sender_sigma = 0.006, receiver_sigma = 0,
                 selection_strength = 1, mutation_rate = 0.005):
        self.num_generations = num_generations
        self.sender_sigma    = sender_sigma
        self.receiver_sigma  = receiver_sigma
        self.selection_strength = selection_strength
        self.mutation_rate = mutation_rate

        self.pb = ProgressBar(self.num_generations-1)
        self.senders = {q: Senders(q, self, population_size = n) for q, n in senders}
        self.receivers = receiver_type(self, population_size = receiver_number)

        self.total_senders = sum([pop.population_size for pop in self.senders.values()])

        for self.i in range(1, self.num_generations):  # Initialize a generation with random values.
            self.update_pop()

    def acceptance_table(self, sender_pop):
        """
        :param Senders sender_pop:
        :return: a 2d array of acceptance values.  d0 = sender #, d1 = receiver #
        """
        gen = self.i - 1
        return np.array([[self.receivers.get_acceptance_individual(sender, receiver)
                        for receiver in self.receivers.strategies]
                        for sender in sender_pop.strategies])

    def get_avg_acceptance(self, q, sender_number):
        return np.mean(self.acceptance_table_dic[q][sender_number])

    def get_aceptees(self, receiver_num):
        """
        :param int receiver_num: The numerical identifier of the receiver
        :return: A dictionary of quality: proportion accepted.
        Note that proportion accepted is out of the total number of senders, not the number of that quality.
        This allows us to weight the sender-population to be unfriendly to the receivers easily.
        Eventually, this should probalby be uncoupled from population size, but it does seem that the population sizes
        are big enough to work.
        """
        return {q: (sum(table[:, receiver_num])/self.total_senders) for q, table in self.acceptance_table_dic.items()}

    def update_pop(self):
        self.acceptance_table_dic = {q: self.acceptance_table(population) for q, population in self.senders.items()}
        for sender in self.senders.values():
            sender.update()
        self.receivers.update()
        if self.i % 100 == 0:
            self.pb.display(self.i)

    def uni_graph(self, ax, gBins, iBins):
        """
        Makes one graph charting two sender types and one receiver type.
        """
        assert len(self.senders) == 2

        high_heatmap, x_edges, y_edges = self.senders[max(self.senders)].sender_heatmap(gBins, iBins)
        high_heatmap = high_heatmap.T

        low_heatmap, _, _ = self.senders[min(self.senders)].sender_heatmap(gBins, iBins)
        low_heatmap = low_heatmap.T

        data_acceptance_profile = self.receivers.get_acceptance_profile(x_edges, y_edges).T

        zipped = np.dstack((low_heatmap, high_heatmap, data_acceptance_profile))
        rgb_values = [[rgb_map(zipped[iBin][gBin]) for gBin in range(gBins)]
                      for iBin in range(iBins)]

        extent = [0, self.num_generations, 0, 1]
        ax.imshow(rgb_values, extent=extent, interpolation='nearest', aspect = 'auto', origin = 'lower')
        ax.ylim(0,1)
        ax.xlim(0, self.num_generations)

    def mean_dist_graph(self, ax, sBins = 100, ycropped = True):
        #Graphs a histogram of the mean strategy played in diff. generations by two sender populations.
        high_points = [np.mean(s_list) for s_list in self.senders[max(self.senders)].strategy_history]
        low_points  = [np.mean(s_list) for s_list in self.senders[min(self.senders)].strategy_history]
        buckets, _, _ = ax.hist([high_points,low_points], bins=sBins, histtype="step", color=["red","blue"])
        ax.axvline(x = .5, c="black")
        if ycropped:
            maximum_high = max(buckets[0])
            ax.set_ylim(0,maximum_high)

    def dist_graph(self, ax, sBins = 100, ycropped = True):
        #Graphs a histogram of the total strategies played over time by two sender populations.
        high_points = self.senders[max(self.senders)].strategy_points()
        low_points  = self.senders[min(self.senders)].strategy_points()
        buckets, _, _ = ax.hist([high_points,low_points], bins=sBins, histtype="step", color=["red","blue"])
        ax.axvline(x = .5, c="black")
        if ycropped:
            maximum_high = max(buckets[0])
            ax.set_ylim(0,maximum_high)


class ProgressBar():
    def __init__(self, goal, width=50):
        self.pointer = 0
        self.width = width
        self.goal = goal

    def display(self, x):
        progress = x/self.goal
        self.pointer = int(self.width*progress)
        display = "|" + "#"*self.pointer + "-"*(self.width-self.pointer)+"|"
        print(display, end='\r')


def varied_simulations(param, value_list):
    s_list = []
    for index, value in enumerate(value_list):
        print(index + 1, len(value_list), sep="/")   # Progress marker
        s = Simulation(**{param: value})
        s_list.append( (s, "{0} = {1}".format(param, value)) )  # adding a tuple of (simulation, graph header)
    print("Saving...")
    fname = "{0}-varied.pik".format(param)
    with open(fname, 'wb') as f:
        pickle.dump(s_list, f, -1)


def vary_attribute_uni_graph(param):
    fname = "{0}-varied.pik".format(param)
    with open(fname, 'rb') as f:
        s_list = pickle.load(f)

    fig, axlist = plt.subplots(len(s_list), 1, sharex=True)
    for index, (s, title) in enumerate(s_list):
        axlist[index].set_title(title)
    plt.show()


def vary_attribute_graph(param, graph_type = "uni_graph"):
    fname = "{0}-varied.pik".format(param)
    with open(fname, 'rb') as f:
        s_list = pickle.load(f)

    assert graph_type in ["uni_graph", "dist_graph", "mean_dist_graph"]
    fig, axlist = plt.subplots(len(s_list), 1, sharex=True)
    for index, (s, title) in enumerate(s_list):
        graphing_method = getattr(s, graph_type)
        graphing_method(axlist[index])
        axlist[index].set_title(title)
    plt.show()


#sender_settings = [((low_qual, 50), (2, 25)) for low_qual in np.arange(.20, 1, .20)]
#varied_simulations("senders", sender_settings)
#varied_simulations("sender_sigma", np.arange(0, .013, .003))
#varied_simulations("receiver_sigma", np.arange(0, .006, .001))

#s = Simulation(num_generations=50000)
# with open("basic.pik", 'wb') as f:
#     pickle.dump(s, f, -1)

#with open("basic.pik", 'rb') as f:
#     s = pickle.load(f)
# s.dist_graph(100)

s = Simulation(num_generations=10000)
# fig, ax = plt.subplots(1, 1, sharex=True)
# s.dist_graph(ax)
# plt.show()

