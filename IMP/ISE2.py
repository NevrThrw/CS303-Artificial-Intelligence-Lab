import sys
import numpy as np
import time
import random
import getopt
import copy

model = ['IC', 'LT']


def get_input():
    options, args = getopt.getopt(sys.argv[1:], 'i:s:m:t:')
    for tu in options:
        if tu[0] == '-i':
            social_network_filepath = tu[1]
        elif tu[0] == '-s':
            seed_set_filepath = tu[1]
        elif tu[0] == '-m':
            model_name = tu[1]
            if model_name not in model:
                raise Exception('No such model !!')
        elif tu[0] == '-t':
            time_budget = int(tu[1])
    social_network = open(social_network_filepath, 'r')
    seed_set = open(seed_set_filepath, 'r')
    return social_network, seed_set, model_name, time_budget


def build_graph(social_network):
    data = social_network.readlines()
    detail = data[0].split(' ')
    node_num = int(detail[0])
    # edge_num = detail[1]
    network = np.zeros((node_num + 1, node_num + 1), dtype=float)
    for i in range(1, len(data)):
        line = data[i].split(' ')
        network[int(line[0])][int(line[1])] = float(line[2])
    return network


def IC_model(network, seed):
    '''

    :param network: the graph of the social network
    :param seed:  the initial active node set in social network
    :return:
    '''
    # the active node set
    active_set = copy.deepcopy(seed)
    # the mark of active node
    active_flag = [0] * len(network)
    count = len(active_set)
    for node in active_set:
        active_flag[node] = 1
    while active_set:
        new_active_set = set()
        for index in active_set:
            for neighbor in range(len(network)):
                if network[index][neighbor] > 0 and active_flag[neighbor] == 0:
                    if random.random() < network[index][neighbor]:
                        new_active_set.add(neighbor)
                        active_flag[neighbor] = 1
        active_set = new_active_set
        count += len(active_set)
    return count


def get_weight(network, node, active_flag):
    weight = 0
    for i in range(len(network)):
        if active_flag[i] == 1:
            weight += network[i][node]
    return weight


def LT_model(network, seed):
    '''

        :param network: the graph of the social network
        :param seed:  the initial active node set in social network
        :return:
    '''
    active_set = copy.deepcopy(seed)
    active_flag = [0] * len(network)
    threshold = [0] * len(network)
    for i in range(len(network)):
        threshold[i] = random.random()
        if threshold[i] == 0 and i not in active_set:
            active_set.append(i)
    for node in active_set:
        active_flag[node] = 1
    count = len(active_set)
    while active_set:
        new_active_set = set()
        for index in active_set:
            for neighbor in range(len(network)):
                if network[index][neighbor] > 0 and active_flag[neighbor] == 0:
                    if threshold[neighbor] <= get_weight(network, neighbor, active_flag):
                        new_active_set.add(neighbor)
                        active_flag[neighbor] = 1
        active_set = new_active_set
        count += len(active_set)
    return count


if __name__ == '__main__':
    start_time = time.time()
    social_network, seed_set, model_name, time_budget = get_input()
    network = build_graph(social_network)
    seed = seed_set.readlines()
    for i in range(len(seed)):
        seed[i] = int(seed[i].replace('\n', ''))
    social_network.close()
    seed_set.close()
    sum, iteration = 0, 0
    for i in range(100000):
        if model_name == 'IC':
            count = IC_model(network, seed)
        elif model_name == 'LT':
            count = LT_model(network, seed)
        sum += count
        iteration += 1
        if time_budget - 3 < time.time() - start_time:
            break
    print(sum / iteration)
