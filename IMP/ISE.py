import sys
import numpy as np
import time
import random
import getopt
import copy


network = list()
edges = dict()
pre_neighbors = dict()
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
    global network
    global edges
    global pre_neighbors
    data = social_network.readlines()
    detail = data[0].split(' ')
    node_num = int(detail[0])
    network = np.zeros((node_num + 1, node_num + 1), dtype=float)
    for i in range(0, len(network)):
        edges[i] = []
        pre_neighbors[i] = []
    for line in data[1:]:
        line = line.replace('\n', '')
        line = line.split(" ")
        fro, to, weight = int(line[0]), int(line[1]), float(line[2])
        network[fro][to] = weight
        edges[fro].append(to)
        pre_neighbors[to].append(fro)


def IC_model(seed):
    active_set = seed
    flag = [0] * len(network)
    for node in active_set:
        flag[node] = 1
    count = 0
    count += len(seed)
    while len(active_set) > 0:
        new_active_set = set()
        for index in active_set:
            # print(edges.get(index))
            for neighbor in edges.get(index):
                if random.random() < network[index][neighbor]:
                    if flag[neighbor] == 0:
                        new_active_set.add(neighbor)
                        flag[neighbor] = 1
        # print(len(new_active_set))
        count += len(new_active_set)
        active_set = new_active_set
    return count


def get_weight(node, ans_set):
    weight = 0
    for i in pre_neighbors.get(node):
        if i in ans_set:
            weight += network[i][node]
    return weight


def LT_model(seed):
    active_set = set(copy.deepcopy(seed))
    threshold = dict()
    ans_set = set()  # total active node
    for node in seed:
        ans_set.add(node)
    while active_set:
        new_active_set = set()
        for index in active_set:
            for neighbor in edges.get(index):
                if neighbor not in ans_set:
                    threshold[neighbor] = threshold[neighbor] if neighbor in threshold.keys() else random.random()
                    if threshold[neighbor] <= get_weight(neighbor, ans_set):
                        new_active_set.add(neighbor)
        active_set = new_active_set
        ans_set = ans_set | active_set
    return len(ans_set)


if __name__ == '__main__':
    start_time = time.time()
    social_network, seed_set, model_name, time_budget = get_input()
    build_graph(social_network)
    seed = seed_set.readlines()
    for i in range(len(seed)):
        seed[i] = int(seed[i].replace('\n', ''))
    # print(seed)
    social_network.close()
    seed_set.close()
    sum, iteration = 0, 0
    for i in range(10000):
        if model_name == 'IC':
            count = IC_model(seed)
            sum += count
        elif model_name == 'LT':
            count = LT_model(seed)
            sum += count
        iteration += 1
        # print(count)
        if time_budget - 3 < time.time() - start_time:
            break
    print(sum / iteration)
