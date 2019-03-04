import sys
import numpy as np
import time
import random
import getopt
import copy

model = ['IC', 'LT']
network = list()
edges = dict()
pre_neighbors = dict()
node = set()


def get_input():
    options, args = getopt.getopt(sys.argv[1:], 'i:k:m:t:')
    for tu in options:
        if tu[0] == '-i':
            social_network_filepath = tu[1]
        elif tu[0] == '-k':
            seed_set_size = int(tu[1])
        elif tu[0] == '-m':
            model_name = tu[1]
            if model_name not in model:
                raise Exception('No such model !!')
        elif tu[0] == '-t':
            time_budget = int(tu[1])
    social_network = open(social_network_filepath, 'r')
    return social_network, seed_set_size, model_name, time_budget


def build_graph(social_network):
    global network
    global edges
    global pre_neighbors
    global node
    data = social_network.readlines()
    edge_set = data[1:]
    detail = data[0].split(' ')
    node_num = int(detail[0])
    network = np.zeros((node_num + 1, node_num + 1), dtype=float)
    for i in range(0, len(network)):
        edges[i] = []
        pre_neighbors[i] = []
    for line in edge_set:
        line = line.replace('\n', '')
        line = line.split(" ")
        fro, to, weight = int(line[0]), int(line[1]), float(line[2])
        network[fro][to] = weight
        edges[fro].append(to)
        pre_neighbors[to].append(fro)
        node.add(fro)


def IC_model(seed, cal=True):
    '''

    :param network: the graph of the social network
    :param seed:  the initial active node set in social network
    :return:
    '''
    # the active node set
    active_set = copy.deepcopy(seed)
    # the mark of active node
    ans_set = set()
    for node in seed:
        ans_set.add(node)
    count = len(active_set)
    while active_set:
        new_active_set = set()
        for index in active_set:
            for neighbor in edges.get(index):
                if random.random() < network[index][neighbor] and neighbor not in ans_set:
                    new_active_set.add(neighbor)
        active_set = new_active_set.copy()
        for node in active_set:
            ans_set.add(node)
        count += len(active_set)
    if cal:
        return count
    else:
        return ans_set


def get_weight(node, ans_set):
    weight = 0
    for i in pre_neighbors.get(node):
        if i in ans_set:
            weight += network[i][node]
    return weight


def LT_model(seed, cal=True):
    '''
        :param network: the graph of the social network
        :param seed:  the initial active node set in social network
        :return:
    '''
    # start = time.time()
    active_set = copy.deepcopy(seed)
    threshold = dict()
    ans_set = set()  # total active node
    for node in seed:
        ans_set.add(node)
    count = len(ans_set)
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
        count += len(active_set)
    # print(time.time()-start)
    if cal:
        return count
    else:
        return ans_set


def getInfulence(seed, model, cal=False):
    if model == 'LT':
        return LT_model(seed, cal)

    elif model == 'IC':
        return IC_model(seed, cal)

    else:
        raise Exception("No such model")


def Lv_CELF(seed_size, model):
    seed_set = set()
    influence = dict()
    flag = dict()
    mg_set = dict()
    num = 0
    for i in node:
        mg_set[i] = getInfulence({i}, model)
        influence[i] = len(mg_set[i])
        flag[i] = 0
    # print('finish initial')
    while len(seed_set) < seed_size and len(influence) > 0:
        hummit = max(influence, key=influence.get)
        # print(hummit, flag[hummit], len(seed_set))
        if flag[hummit] == len(seed_set):
            seed_set.add(hummit)
            num +=influence[hummit]
            influence.pop(hummit)
            for tup in mg_set[hummit]:
                if tup in influence.keys():
                    influence.pop(tup)
        else:
            list2 = getInfulence(seed_set, model)
            list1 = getInfulence(seed_set | {hummit}, model)
            mg_set[hummit] = list1 - list2
            influence[hummit] = len(mg_set[hummit])
            flag[hummit] = len(seed_set)
    # print("end")
    return list(seed_set),num


# def getInfluence2(seed, model, cal=True):
#     if model == 'LT':
#         it = 50
#         sum = 0
#         for i in range(it):
#             sum += LT_model(list(seed), cal)
#         return sum / it
#     elif model == 'IC':
#         it = 50
#         sum = 0
#         for i in range(it):
#             sum += IC_model(list(seed), cal)
#         return sum / it
#     else:
#         raise Exception("No such model")


# def getKey(item):
#     return item[1]


# def CELF(seed_size, model):
#     seed_set = set()
#     influence = dict()
#     flag = dict()
#     for i in node:
#         # print(i)
#         influence[i] = getInfluence2({i}, model)
#         flag[i] = 0
#
#     while len(seed_set) < seed_size:
#         hummit = max(influence, key=influence.get)
#         if flag.get(hummit) == len(seed_set):
#             seed_set.add(hummit)
#             influence.pop(hummit)
#             flag.pop(hummit)
#         else:
#             influence[hummit] = getInfluence2(seed_set | {hummit}, model) - getInfluence2(seed_set, model)
#             flag[hummit] = len(seed_set)
#
#     return list(seed_set)


# def greedy(network, seed_size, model):
#     seed_set = set()
#     while len(seed_set) < seed_size:
#         argmax = 0
#         target = -1
#         for node in range(1, len(network)):
#             temp_set = copy.deepcopy(seed_set)
#             influence1 = getInfulence(network, list(temp_set), model, cal=True)
#             temp_set.add(node)
#             influence2 = getInfulence(network, list(temp_set), model, cal=True)
#             dif = influence2 - influence1
#             if dif > argmax:
#                 argmax = dif
#                 target = node
#         seed_set.add(target)
#     return seed_set

def con(seed, model):
    iteration = 50
    count = 0
    for i in range(iteration):
        if model == "IC":
            count += IC_model(seed)
        elif model == "LT":
            count += LT_model(seed)
    return count / iteration


if __name__ == '__main__':
    start_time = time.time()
    social_network, seed_size, model_name, time_budget = get_input()
    build_graph(social_network)
    social_network.close()
    start = time.time()
    ans_set,nodes = Lv_CELF(seed_size, model_name)
    # ans_set = greedy(seed_size, model_name)
    # ans_set = CELF(seed_size, model_name)
    best_set = ans_set
    best_count = con(best_set,model_name)
    # print(best_count)
    run = time.time() - start
    num = 500
    while num > 0 and time.time() - start_time + run + 1 < time_budget - 2:
        # print(num)/
        temp_ans,temp_nodes = Lv_CELF(seed_size, model_name)
        count = con(temp_ans, model_name)
        num -= 1
        if count > best_count:
            best_set = temp_ans
            best_count = count
            # print(best_count)
    for node in best_set:
        print(node)
    # print(best_count)
    # print(time.time() - start_time)
