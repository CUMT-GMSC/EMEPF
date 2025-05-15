from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
from graph import *
import pickle
from igraph import Graph
import os


def loadData2List(edges_file):
    file = edges_file
    f = open(file ,mode='r')
    line = f.readline()
    edge_list = []
    while line:
        line = line.split()
        line = list(map(int, line))
        edge_list.append(line)
        line = f.readline()
    f.close()
    return edge_list

def edgeList2Graph(edge_list):
    G = nx.DiGraph()
    Nodes = {}
    n = len(edge_list)
    for i in range(n):
        for j in range(2):
            if edge_list[i][j] not in Nodes:
                Nodes[edge_list[i][j]] = Node(edge_list[i][j])
        G.add_edge(Nodes[edge_list[i][0]] ,Nodes[edge_list[i][1]])
    return G ,Nodes


# 读的community数据，第一列是节点，第二列是节点所在的community
# def load_communities(community_file):
#     file = community_file
#     f = open(file ,mode='r')
#     line = f.readline()
#     communities = {}
#     node_community_dic = {}
#     while line:
#         line = line.split()
#         line = list(map(int, line))
#         node_community_dic[line[0]] = line[1] + 1       # 生成  nodeID-cID 的字典
#         if line[1 ] +1 not in communities:
#             community = Community(line[1 ] +1 ,[])        # 如果这是一个新社区，那么new一个社区，社区编号为line[1]+1（社区编号从1开始），社区内成员先初始化为 空 ，
#             communities[line[1 ] +1] = community         # 将这个社区 接入 communities字典 里，生成 cID-社区对象 的字典
#         communities[line[1 ] +1].community.append(line[0])      # 在该社区新加一个节点
#         communities[line[1 ] +1].size += 1               # 该社区的规模 +1
#         line = f.readline()
#     C = Communities(communities ,node_community_dic)
#     f.close()
#     return C

def load_communities(community_file):
    file = community_file
    f = open(file ,mode='r')
    line = f.readline()
    communities = {}
    node_community_dic = {}
    while line:
        line = line.split()
        line = list(map(int, line))
        node_community_dic[line[0]] = line[1]      # 生成  nodeID-cID 的字典
        if line[1 ] not in communities:
            community = Community(line[1 ] ,[])        # 如果这是一个新社区，那么new一个社区，社区编号为line[1]+1（社区编号从1开始），社区内成员先初始化为 空 ，
            communities[line[1 ]] = community         # 将这个社区 接入 communities字典 里，生成 cID-社区对象 的字典
        communities[line[1 ]].community.append(line[0])      # 在该社区新加一个节点
        communities[line[1 ]].size += 1               # 该社区的规模 +1
        line = f.readline()
    C = Communities(communities ,node_community_dic)
    f.close()
    return C

def load_graph(edges_file ,community_file):
    edge_list = loadData2List(edges_file)
    G, Nodes = edgeList2Graph(edge_list)
    C = load_communities(community_file)
    return edge_list, graph(G, Nodes, C)


def load_seed_sets(seed_sets_path):
    seed_sets = []
    with open(seed_sets_path, 'r') as f:
        line = f.readline()
        i = 0
        seed_set = []
        while line:
            i += 1
            seed_set.append(int(line))
            if i == 10:
                seed_sets.append(seed_set)
                i = 0
                seed_set = []
            line = f.readline()
    return seed_sets


def load_graph_based_on_igraph(data_path, selected_community):
    g = Graph.Read_GML(data_path)
    edge_list = [[edge[0]+1, edge[1]+1] for edge in g.get_edgelist()]

    communities = {}
    node_community_dic = {}
    cID_dic = {}  # {community_value:cID}   {'male':0, 'female':1}

    for v in g.vs:
        community_value = v[selected_community]
        if community_value not in cID_dic:
            cID_dic[community_value] = len(cID_dic) + 1
        cID = cID_dic[community_value]
        if cID not in communities:
            communities[cID] = Community(cID, [])
        communities[cID].community.append(v.index + 1)
        communities[cID].size += 1
        node_community_dic[v.index + 1] = cID

    G, Nodes = edgeList2Graph(edge_list)
    C = Communities(communities, node_community_dic)

    return edge_list, graph(G, Nodes, C)



