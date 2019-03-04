from getopt import *
from time import *
from numpy import *
import sys
from random import *
import copy

# Read the network file
def network_reader(file_path):
    line_one = open(file_path).readline()
    ovr_data = str.split(line_one)
    vertex_num = int(ovr_data[0])
    edge_num = int(ovr_data[1])
    graph_edge = loadtxt(file_path, skiprows=1)
    return vertex_num, edge_num, graph_edge


# Read the seed file
def seed_reader(file_path):
    seeds = set()
    lines = open(file_path).readlines()
    for line in lines:
        seeds.add(int(line.split()[0]))
    return seeds


class Graph:
    nodes = set()
    edges = []
    in_edges = []
    weight = {}

    def __init__(self, numpy_array, num_vertex):
        for i in range(0, num_vertex):
            self.add_node(i)
        for i in range(0, len(numpy_array)):
            self.add_edge(numpy_array[i][0], numpy_array[i][1], numpy_array[i][2])

    def add_node(self, value):
        self.edges.append([])
        self.in_edges.append([])

    def add_edge(self, from_node, to_node, weight):
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edges[int(from_node) - 1].append(int(to_node) - 1)
        self.in_edges[int(to_node) - 1].append(int(from_node) - 1)
        self.weight[(int(from_node) - 1, int(to_node) - 1)] = weight

def ic(seed):
    work=seed.copy()
    all=seed.copy()
    num=0
    num+=len(work)
    while len(work)>0:
        # print(num)
        new_work=set()
        for node in work:
            for near_node in graph.edges[node-1]:
                if random() < graph.weight[node-1,near_node]:
                    if near_node+1 not in all:
                        all.add(near_node+1)
                        new_work.add(near_node+1)
        # print(len(new_work))
        num+=len(new_work)
        work=new_work.copy()
    return num

def it(seed):
    work=seed.copy()
    all=seed.copy()
    ran={}

    # t=time()
    while work:
        new_work=set()
        for node in work:
            for near_node in graph.edges[node-1]:
                if near_node + 1 not in all:
                    tol_weight = 0
                    for n in graph.in_edges[near_node]:
                        if n + 1 in all:
                            tol_weight = tol_weight + graph.weight[(n, near_node)]
                    if near_node not in ran:
                        ran[near_node]=random()
                    if tol_weight > ran[near_node]:
                            new_work.add(near_node + 1)
                            all.add(near_node + 1)
        work=new_work.copy()
    return len(all)


def spread_check(graph, seed, model):
    global R
    sum = 0.0
    R=100
    if model == "IC":
        for j in range(0, R):
            count = ic( seed)
            sum = count + sum
            # print(count)
    else:
        for j in range(0, R):
            count = it( seed)
            sum = count + sum
            # print(count)
    return sum / float(R)

def read_cmd():
    try:
        opts, agrs =getopt(sys.argv[0:],'i:s:m:t')
    except:
        print("wrong in read cmd")
        sys.exit(2)
    graph_path=agrs[2]
    seed_path=agrs[4]
    model=agrs[6]
    time=int(agrs[8])
    return graph_path,seed_path,model,time

def main():
    # file_name=""
    # graph_path, seed_path, model, time=read_cmd()
    graph_path='network2.txt'
    seed_path='seeds4.txt'
    model='IC'

    vertex_num, edge_num, graph_edge=network_reader(graph_path)
    seed=seed_reader(seed_path)
    global graph
    graph=Graph(graph_edge,vertex_num)
    # seed={17, 23, 28, 32, 36, 39, 44, 45, 43, 48, 50, 51, 52, 53, 56, 58, 57, 60, 61, 62, 59}#{48, 52, 53, 56, 58}
    result=spread_check(graph,seed,model)
    print(result)

if __name__ == '__main__':
    main()