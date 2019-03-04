'''
constraint: 1,each route must start at v0 and end at v0
            2,total serviced demand on a route must not exceed Q
            3,each task should be serviced
graph: the matrix of the whole graph consist of all nodes, the value in each cell: if -1 means not connected,
                                                                                   if 0 means connected but not a demand
                                                                                   if >0 means connected and the demand is the value in the cell
edge_cost: a matrix store the cost of the edges in graph, if two nodes are not connected, the cost will be 9999999
min_cost: the matrix store the shortest cost from one node to the other
choose_rule: the rule that use to pick task to build a initial sequence.In this case, we use when vehicle is not half-full,
maximize the distance, else minimize the distance
'''

import numpy
import sys
import getopt
import multiprocessing
import random
import copy
import time
import math


def get_graph(filename):
    fil = open(filename)
    file = fil.readlines()
    # get vertices
    vert = file[1].split(':')
    vert[1] = vert[1].replace('\n', '')
    vertices = int(vert[1])
    # get depot
    depot = file[2].split(':')
    depot[1] = depot[1].replace('\n', '')
    depot = int(depot[1])
    # get demand edges
    demand_edge = file[3].split(':')
    demand_edge[1] = demand_edge[1].replace('\n', '')
    demand = int(demand_edge[1])
    # get vehicles
    vech = file[5].split(':')
    vech[1] = vech[1].replace('\n', '')
    vehicles = int(vech[1])
    # get capacity
    cap = file[6].split(':')
    cap[1] = cap[1].replace('\n', '')
    capacity = int(cap[1])
    # build the graph
    graph = numpy.ones((vertices + 1, vertices + 1))
    graph = graph * -1
    edge_cost = numpy.ones((vertices + 1, vertices + 1))
    edge_cost = edge_cost * 9999999
    for i in range(1, vertices + 1):
        edge_cost[i][i] = 0
    min_cost = numpy.zeros((vertices + 1, vertices + 1))
    i = 9
    while file[i] != 'END':
        line = file[i].split()
        graph[int(line[0])][int(line[1])] = int(line[3])
        graph[int(line[1])][int(line[0])] = int(line[3])
        edge_cost[int(line[0])][int(line[1])] = int(line[2])
        edge_cost[int(line[1])][int(line[0])] = int(line[2])
        i += 1
    min_cost = get_min_cost(min_cost, edge_cost, vertices)
    fil.close()
    return vertices, depot, demand, vehicles, capacity, graph, edge_cost, min_cost


def get_input():
    filename = sys.argv[1]
    vertices, depot, demand, vehicles, capacity, graph, edge_cost, min_cost = get_graph(
        filename)
    options, args = getopt.getopt(sys.argv[2:], 't:s:')
    for tup in options:
        if tup[0] == '-t':
            terminate_time = int(tup[1])
        if tup[0] == '-s':
            random_seed = int(tup[1])
    return vertices, depot, demand, vehicles, capacity, graph, edge_cost, min_cost, terminate_time, random_seed


def dijkstra(p, min_cost, edge_cost, vertices):
    visited = [0] * (vertices + 1)
    visited[p] = 1
    dis = [9999999] * (vertices + 1)
    for i in range(1, vertices + 1):
        dis[i] = edge_cost[p][i]
    for i in range(vertices - 1):
        mini = 9999999
        u = 0
        for j in range(1, vertices + 1):
            if visited[j] == 0 and dis[j] < mini:
                mini = dis[j]
                u = j
        visited[u] = 1
        for j in range(1, vertices + 1):
            if edge_cost[u][j] < 9999999:
                if dis[j] > dis[u] + edge_cost[u][j]:
                    dis[j] = dis[u] + edge_cost[u][j]
    for i in range(1, vertices + 1):
        min_cost[p][i] = dis[i]


def get_min_cost(min_cost, edge_cost, vertices):
    for i in range(1, vertices + 1):
        dijkstra(i, min_cost, edge_cost, vertices)
    return min_cost


def path_scanning(t, depot, graph, edge_cost, min_cost, vehicles, vertices, capacity):
    solu = [[]]
    temp_graph = copy.deepcopy(graph)
    arcs = get_free_arcs(temp_graph, vertices)
    for w in range(1, vehicles + 1):
        pos = depot
        cap = capacity
        rout = []
        while True:
            dis = 9999999
            for road in arcs:
                if cap - road[2] < 0:
                    continue
                dis2 = min_cost[pos][road[0]]
                if t == 1:
                    if dis2 < dis:
                        dis = dis2
                        u = road
                    elif dis2 == dis:
                        if road[2] / edge_cost[road[0]][road[1]] > u[2] / edge_cost[u[0]][u[1]]:
                            u = road
                elif t == 2:
                    if dis2 < dis:
                        dis = dis2
                        u = road
                    elif dis2 == dis:
                        if min_cost[road[1]][depot] < min_cost[u[1]][depot]:
                            u = road
                elif t == 3:
                    if dis2 < dis:
                        dis = dis2
                        u = road
                    elif dis2 == dis:
                        if edge_cost[road[0]][road[1]] < edge_cost[u[0]][u[1]]:
                            u = road
                elif t == 4:
                    if dis2 < dis:
                        dis = dis2
                        u = road
                    elif dis2 == dis:
                        if min_cost[road[1]][depot] > min_cost[u[1]][depot]:
                            u = road
                elif t == 5:
                    if dis2 < dis:
                        dis = dis2
                        u = road
                    elif dis2 == dis:
                        if cap > capacity / 2 and min_cost[road[1]][depot] > min_cost[u[1]][depot]:
                            u = road
                        elif cap < capacity / 2 and min_cost[road[1]][depot] < min_cost[u[1]][depot]:
                            u = road
                elif t == 6:
                    if dis2 < dis:
                        dis = dis2
                        u = road
            if dis == 9999999:
                break
            arcs.remove(u)
            arcs.remove((u[1], u[0], u[2]))
            rout.append((u[0], u[1]))
            pos = u[1]
            cap -= u[2]
        solu.append(rout)
    return solu


def get_free_arcs(gra, vertices):
    arcs = []
    for i in range(vertices + 1):
        for j in range(i + 1, vertices + 1):
            if gra[i][j] > 0:
                arcs.append((i, j, gra[i][j]))
                arcs.append((j, i, gra[j][i]))
    return arcs


def fitness(road, depot, edge, mini):
    cost = 0
    for i in range(1, len(road)):
        last = 1
        for tu in road[i]:
            cost += edge[tu[0]][tu[1]]  # the cost of demand edge
            cost += mini[last][tu[0]]  # the cost of moving to next demand edge
            last = tu[1]
        cost += mini[last][depot]
    return cost


def print_result(solut):
    result = []
    for i in range(1, len(solut)):
        result.append(0)
        for tu in solut[i]:
            result.append(tu)
        result.append(0)
    res = str(result).replace('[', '')
    res = res.replace(']', '')
    res = res.replace(' ', '')
    print('s ' + res)


def find_best(route_list, depot, edge, mini):
    cost = 9999999
    for rout in route_list:
        cost1 = fitness(rout, depot, edge, mini)
        if cost1 < cost:
            cost = cost1
            best_route = rout
    return best_route, cost


# check whether the answer satisfy the three constraints
def check_valid(road, dem, gra, cap):
    ro_list = []
    for i in range(1, len(road)):
        cost = 0
        for ro in road[i]:
            cost += gra[ro[0]][ro[1]]
            ro_list.append(ro)
        if cost > cap:  # not satisfy the sum of demand on a road should be smaller or equals to capacity Q
            return False
    if len(set(ro_list)) != dem:  # not satisfy all the demands are serviced
        return False
    return True


def PS_for_MS(o, sub_solution, depot, edge_cost, min_cost, graph, capacity):
    free_arcs = []
    new_solution = []
    for t in sub_solution:
        for arc in t:
            free_arcs.append(arc)
            free_arcs.append((arc[1], arc[0]))
    while free_arcs:
        pos = depot
        cap = capacity
        new_sub_sloution = []
        while True:
            dis = 9999999
            for arc in free_arcs:
                if cap - graph[arc[0]][arc[1]] < 0:
                    continue
                dis2 = min_cost[pos][arc[0]]
                if o == 1:
                    if dis2 < dis:
                        dis = dis2
                        u = arc
                    elif dis2 == dis:
                        if cap > capacity / 2 and min_cost[arc[0]][depot] > min_cost[u[0]][depot]:
                            u = arc
                        elif cap < capacity / 2 and min_cost[arc[0]][depot] < min_cost[u[0]][depot]:
                            u = arc
                elif o == 2:
                    if dis2 < dis:
                        dis = dis2
                        u = arc
                    elif dis2 == dis:
                        if min_cost[arc[0]][depot] < min_cost[u[0]][depot]:
                            u = arc
                elif o == 3:
                    if dis2 < dis:
                        dis = dis2
                        u = arc
                    elif dis2 == dis:
                        if graph[arc[0]][arc[1]] / edge_cost[arc[0]][arc[1]] < graph[u[0]][u[1]] / edge_cost[u[0]][
                            u[1]]:
                            u = arc
                elif o == 4:
                    if dis2 < dis:
                        dis = dis2
                        u = arc
                    elif dis2 == dis:
                        if min_cost[arc[0]][depot] > min_cost[u[0]][depot]:
                            u = arc
                elif o == 5:
                    if dis2 < dis:
                        dis = dis2
                        u = arc
                    elif dis2 == dis:
                        if cap > capacity / 2 and min_cost[arc[1]][depot] > min_cost[u[1]][depot]:
                            u = arc
                        elif cap < capacity / 2 and min_cost[arc[1]][depot] < min_cost[u[1]][depot]:
                            u = arc
            if dis == 9999999:
                break
            free_arcs.remove(u)
            free_arcs.remove((u[1], u[0]))
            new_sub_sloution.append(u)
            pos = u[1]
            cap -= graph[u[0]][u[1]]
        new_solution.append(new_sub_sloution)
    return new_solution


def MS(solution, sub_pos, depot, edge, graph, mini, capacity):
    sub_solution = []
    for t in range(len(sub_pos)):
        sub_solution.append(solution[sub_pos[t]])
    new_solution = []
    for o in range(5):
        new_sub_sloution = PS_for_MS(
            (o % 5 + 1), sub_solution, depot, edge, mini, graph, capacity)
        newer = copy.deepcopy(solution)
        for t in range(len(sub_pos)):
            newer.remove(solution[sub_pos[t]])
        newer += new_sub_sloution
        new_solution.append(newer)
    new_best, new_cost = find_best(new_solution, depot, edge, mini)
    return new_best


def swap(road, veh, dem, gr, edge, mini, cap, depot, t_time, start_time):
    t = len(road)
    result = []
    for i in range(1, t):
        if t_time < time.time() - start_time + 2:
            break
        for j in range(len(road[i])):
            if t_time < time.time() - start_time + 2:
                break
            node1 = road[i][j]
            for o in range(i, t):
                if t_time < time.time() - start_time + 2:
                    break
                st = j + 1 if o == i else 0
                for p in range(st, len(road[o])):
                    if t_time < time.time() - start_time + 2:
                        break
                    road2 = copy.deepcopy(road)
                    temp = road[o][p]
                    road2[o][p] = node1
                    road2[i][j] = temp
                    if check_valid(road2, dem, gr, cap):
                        result.append(road2)

    if not result:
        return road
    else:
        return find_best(result, depot, edge, mini)[0]


def insertion(road, veh, dem, gr, edge, mini, cap, depot, t_time, start_time):
    # insertion all for one
    t = len(road)
    result = []
    for i in range(1, t):
        if t_time < time.time() - start_time + 2:
            break
        for j in range(len(road[i])):
            if t_time < time.time() - start_time + 2:
                break
            road3 = copy.deepcopy(road)
            node1 = road3[i][j]
            road3[i].remove(node1)
            for o in range(1, t):
                if t_time < time.time() - start_time + 2:
                    break
                for p in range(len(road3[o])):
                    if t_time < time.time() - start_time + 2:
                        break
                    road4 = copy.deepcopy(road3)
                    road4[o].insert(p, node1)
                    if check_valid(road4, dem, gr, cap):
                        result.append(road4)
    if not result:
        return road
    else:
        return find_best(result, depot, edge, mini)[0]


def insertion2(road, veh, dem, gr, edge, mini, cap, depot, t_time, start_time):
    t = len(road)
    result = []
    for i in range(1, t):
        if t_time < time.time() - start_time + 2:
            break
        for j in range(len(road[i]) - 1):
            if t_time < time.time() - start_time + 2:
                break
            road5 = copy.deepcopy(road)
            node1 = road5[i][j]
            node2 = road5[i][j + 1]
            road5[i].remove(node1)
            road5[i].remove(node2)
            for o in range(i, t):
                if t_time < time.time() - start_time + 2:
                    break
                for p in range(len(road5[o])):
                    if t_time < time.time() - start_time + 2:
                        break
                    road6 = copy.deepcopy(road5)
                    road6[o].insert(p + 1, node1)
                    road6[o].insert(p + 2, node2)
                    if check_valid(road6, dem, gr, cap):
                        result.append(road6)
    if not result:
        return road
    else:
        return find_best(result, depot, edge, mini)[0]


def Merge_Split(rou, veh, dem, gr, edge, mini, cap, depot, t_time, start_time):
    result = []
    t = len(rou)
    num = int((t - 1) / 2)
    for i in range(1, t - num):
        if t_time < time.time() - start_time + 2:
            break
        pos = []
        for j in range(num):
            if t_time < time.time() - start_time + 2:
                break
            pos.append(i + j)
            new_solution = MS(rou, pos, depot, edge, gr, mini, cap)
            if check_valid(new_solution, dem, gr, cap):
                result.append(new_solution)
    if not result:
        return rou
    else:
        return find_best(result, depot, edge, mini)[0]


def sub_thread(i, depot, loacl_list, dem, veh, gr, edge, mini, t_time, start_time, cap):
    t = fitness(loacl_list[i], depot, edge, mini)
    temperature = t * t
    rou = loacl_list[i]  # update the best solution
    best = rou
    check = 0
    new_solution = []

    time1 = time.time()
    new_solution.append(insertion(rou, veh, dem, gr, edge, mini, cap, depot, t_time, start_time))
    time2 = time.time()
    insert1_time = time2 - time1
    if insert1_time > check:
        check = insert1_time
    time1 = time.time()
    new_solution.append(swap(rou, veh, dem, gr, edge, mini, cap, depot, t_time, start_time))
    time2 = time.time()
    swap_time = time2 - time1
    if swap_time > check:
        check = swap_time
    time1 = time.time()
    new_solution.append(insertion2(rou, veh, dem, gr, edge, mini, cap, depot, t_time, start_time))
    time2 = time.time()
    insert2_time = time2 - time1
    if insert2_time > check:
        check = insert2_time
    time1 = time.time()
    new_solution.append(Merge_Split(rou, veh, dem, gr, edge, mini, cap, depot, t_time, start_time))
    time2 = time.time()
    MS_time = time2 - time1
    if MS_time > check:
        check = MS_time
    newer = find_best(new_solution, depot, edge, mini)[0]
    a = fitness(newer, depot, edge, mini)
    b = fitness(rou, depot, edge, mini)
    if a < b:
        rou = newer
    break_time = 8
    tim = 0
    while temperature > 0.01 and t_time > time.time() - start_time + check + 3:
        if tim >= break_time:
            break
        for j in range(4):  # the length of each step
            new_solution = []
            if t_time > time.time() - start_time + swap_time and j == 0:
                new_solution.append(swap(rou, veh, dem, gr, edge, mini, cap, depot, t_time, start_time))
            if t_time > time.time() - start_time + insert1_time and j == 1:
                new_solution.append(insertion(rou, veh, dem, gr, edge, mini, cap, depot, t_time, start_time))
            if t_time > time.time() - start_time + insert2_time and j == 2:
                new_solution.append(insertion2(rou, veh, dem, gr, edge, mini, cap, depot, t_time, start_time))
            if t_time > time.time() - start_time + MS_time and j == 3:
                new_solution.append(Merge_Split(rou, veh, dem, gr, edge, mini, cap, depot, t_time, start_time))
            if not new_solution:
                break
            new_route = find_best(new_solution, depot, edge, mini)[0]
            a = fitness(new_route, depot, edge, mini)
            b = fitness(rou, depot, edge, mini)
            if a < b:
                rou = new_route
            elif math.exp(float((b - a) / temperature)) > numpy.random.uniform(0.0, 1.0):  # a>=b
                rou = new_route
        # print('best', fitness(best, depot, edge, mini), 'rou', fitness(rou, depot, edge, mini))
        if fitness(rou, depot, edge, mini) < fitness(best, depot, edge, mini):
            best = rou
            loacl_list[i] = best  # update the best solution
            tim = 0
        else:
            tim += 1
        temperature = temperature * 0.85
        # print(temperature)


def main():
	start = time.time()
    vertices, depot, demand, vehicles, capacity, graph, edge_cost, min_cost, terminate_time, random_seed = get_input()
    solution_set = []
    random.seed(random_seed)
    manager = multiprocessing.Manager()
    #pool = multiprocessing.Pool()
    num = 8
    local_list = manager.list(range(num))
    for i in range(num):
        solution_set.append(path_scanning(
            (i % 6 + 1), depot, graph, edge_cost, min_cost, vehicles, vertices, capacity))
	for i in range(num):
        local_list[i] = solution_set[i]

    processes = []
    for i in range(8):
        sub_process = multiprocessing.Process(target=sub_thread,
                                              args=(
                                                  i, depot, local_list, demand, vehicles, graph, edge_cost,
                                                  min_cost, terminate_time, start, capacity))
        processes.append(sub_process)
        sub_process.start()
    for cess in processes:
        cess.join()
	
    best_solution, best_cost = find_best(
        local_list, depot, edge_cost, min_cost)
    print_result(best_solution)
    print('q', int(best_cost))


if __name__ == '__main__':
    # start = time.time()
    main()
    # print(time.time() - start)
