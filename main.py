# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy

import networkx
import networkx as nx
import numpy as np
import openpyxl
from EMO import emo
from LoadData import *
from draw_graphs import DrawGraph
from graph import *
from CFF_model import *
from Measure import *
from saveData import *
import pickle
from Measure import *
import pandas as pd
import os


def multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, N, communities, probability, graph_type):
    Q = Manager().Queue()
    process_list = []
    RRs_set = []

    G, Nodes = edgeList2Graph(edge_list)
    GRAPH = graph(G, Nodes, communities)
    imm = IMM(GRAPH, k, approx_rate, err_pr, graph_type)
    imm.set_edge_prob(probability)
    for i in range(N):
        # G, Nodes = edgeList2Graph(edge_list)
        # GRAPH = graph(G, Nodes, communities)
        # imm = IMM(GRAPH, k, approx_rate, err_pr, graph_type)
        # imm.set_edge_prob(probability)
        p = Process(target=imm.single_fair_Sampling, args=(Q, math.ceil((RRs_num / imm.graph.nodes_number) / N)))
        process_list.append(p)
        nodes_number = imm.graph.nodes_number
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()
    while Q.qsize() != 0:
        RRs = Q.get_nowait()
        RRs_set += RRs
    for index, RRs in enumerate(RRs_set):
        RRs.insert(0, index)
    return RRs_set


def main_EMO_email():
    k = 10
    ps = [0.01]
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 8
    RRs_num = 294938
    graph_type = 1
    attribute = 'department'
    file = 'data/email/email-Eu-core.txt'
    community_file = 'data/email/email-Eu-core-department-labels.txt'
    c_ps = [0.75]
    m_ps = [0.2]
    fc_ratios = [0.7] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # set to 0.2
    fm_ps = [0.2] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # set to 0.7
    N = 30
    T = 30
    for p in ps:
        edge_list, G = load_graph(file, community_file)
        communities = G.Communities
        for key in communities.communities:
            community = communities.communities[key]
        cff = IMM(G, k, approx_rate, err_pr, graph_type)
        cff.set_edge_prob(p)

        RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core,
                                     communities, p,
                                     graph_type)

        result_folder = 'zjx_results/email_emo/'
        # seed_sets_path = 'seed_set/seed_DD.txt'
        # seed_sets = load_seed_sets(seed_sets_path)

        seed_De = load_seed_sets('zjx_results/email_result/De.txt')
        seed_DD = load_seed_sets('zjx_results/email_result/DD.txt')
        seed_IMM = load_seed_sets('zjx_results/email_result/IMM_p=0.01_k=10.txt')
        seed_DC = load_seed_sets('zjx_results/email_result/DC_p=0.01_k=10.txt')
        seed_CFF = load_seed_sets('zjx_results/email_result/CFF_p=0.01_k=10.txt')
        seed_WF = load_seed_sets('zjx_results/email_result/WF.txt')

        # seed_combine = [seed_De[0]] + [seed_DD[0]] + [seed_IMM[0]] + [seed_DC[0]] + [seed_CFF[0]] + [seed_WF[0]]
        # seed_combine = [seed_De[0]] + [seed_DD[0]] + [seed_IMM[0]] +  [seed_CFF[0]] + [seed_WF[0]]
        seed_combine = [[122, 937,  22,  64,  66,  82, 510, 161, 795, 210],
                        [222, 766,  22,  64,  66, 435, 108, 161,  94, 380],
                        [233, 906,  22,  64,  66, 384, 108, 161, 965, 474],
                        [238,   1,  22,  64,  66, 257, 108, 161, 230, 104],
                        [238, 236,  22,  64,  66,  87, 108, 161, 143, 134],
                        [374, 236,  22,  64,  66,  87, 108, 161, 143, 134],
                        [406,   1,  22,  64, 126, 781, 108, 115, 207, 474],
                        [482,   1,  22,  64, 872,  82, 108, 161, 795, 134],
                        [482, 185,  22,  64, 292, 534, 108, 161, 143, 210],
                        [648, 185,  22,  64,  66, 435, 108, 161,  13, 210],
                        [648, 185,  22,  64, 222, 169, 914, 115,  94, 210],
                        [720,   1,  22,  64, 222,  82, 914, 115, 795, 134],
                        [720,   1,  22,  64, 293,  10, 108, 115, 475, 134],
                        [720,   1,  22,  64, 511, 612, 108, 115, 795, 474],
                        [720,   1,  22,  64, 843, 612, 108, 115, 134, 474],
                        [720,  44,  22,  64,  66,  87, 108, 161, 143, 134],
                        [773, 236,  22,  64,  66, 435, 108, 161,  30, 134],
                        [161, 83, 122, 250, 184, 108, 14, 378, 87, 212],
                        [161, 184, 108, 63, 250, 257, 64, 378, 22, 366]]

        for c_p in c_ps:
            for m_p in m_ps:
                for fc_ratio in fc_ratios:
                    for fm_p in fm_ps:
                        result_file = 'c_p=' + str(c_p) + ' m_p=' + str(m_p) + ' fc_ratio=' + str(
                            fc_ratio) + ' fm_p=' + str(fm_p) + ' N=' + str(N) + ' T=' + str(T) + '.json'
                        mo = emo(cff, c_p, m_p, fc_ratio, fm_p, N, 2, k, T, [1, -1], result_folder + result_file,
                                 'email_result', attribute, edge_list, cpu_core, 'RFMFIM')
                        mo.fair_nsga2(RRs_set, seed_combine)
                        # mo.fair_nsga2(RRs_set, seed_DD)


def main_CFF_networksgml():
    # only run function 2: return 1-math.exp(-function_parameter*x)
    ps = [0.1]
    RRs_num = 253858
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 1
    concave_function_types = [2]
    function_parameters_dic = {
        2: [40],
    }
    # ks = [5, 10, 15, 20]
    ks = [10]
    file_path = "data/networks_gml"
    # selected_communities = ["gender", "ethnicity", "age"]
    selected_communities = ["ethnicity"]
    for p in ps:
        for selected_community in selected_communities:
            for k in ks:
                # for gml_number in range(24):
                for gml_number in range(1):
                    edge_list, G = load_graph_based_on_igraph(file_path + '/graph_spa_500_' + str(gml_number) + '.gml',
                                                              selected_community)
                    communities = G.Communities
                    cff = IMM(G, k, approx_rate, err_pr, graph_type)
                    cff.set_edge_prob(p)
                    RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                                 graph_type)
                    for concave_function_type in concave_function_types:
                        # print('concave_function_type=', concave_function_type)
                        for function_parameter in function_parameters_dic[concave_function_type]:
                            # print('function_parameter=',function_parameter)
                            seed_set = cff.CELF_community_fair_NodeSelection(RRs_set, concave_function_type,
                                                                             function_parameter)

                            # save_result(
                            #     'CFF_result/' + 'p=' + str(p) + ' attr=' + str(selected_community) + ' k=' + str(
                            #         k) + '.json',
                            #     gml_number, seed_set)
                            # print(
                            #     'p=' + str(p) + ' attr=' + str(selected_community) + ' k=' + str(k) + ' gml_num=' + str(
                            #         gml_number))
                            print(f'seed_set: {seed_set}')
                            with open('zjx_results/gml_result/CFF_p=0.1_k=10_RRs_253858.txt', 'w') as f:
                                f.write(str(seed_set))


def main_IMM_networkgml():
    ps = [0.1]
    RRs_num = 253858
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 1
    # ks = [5, 10, 15, 20]
    ks = [10]
    file_path = "data/networks_gml"
    # selected_communities = ["gender", "ethnicity", "age"]
    selected_communities = ["ethnicity"]
    for p in ps:
        for selected_community in selected_communities:
            for k in ks:
                # for gml_number in range(24):
                for gml_number in range(1):
                    edge_list, G = load_graph_based_on_igraph(file_path + '/graph_spa_500_' + str(gml_number) + '.gml',
                                                              selected_community)
                    communities = G.Communities
                    cff = IMM(G, k, approx_rate, err_pr, graph_type)
                    cff.set_edge_prob(p)
                    RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                                 graph_type)
                    seed_set = cff.NodeSelection(RRs_set)
                    print(f'seed_set: {seed_set}')
                    with open('zjx_results/gml_result/IMM_p=0.1_k=10_RRs_253858.txt', 'w') as f:
                        f.write(str(seed_set))
                    # save_result(
                    #     'IMM_result/' + 'p=' + str(p) + ' attr=' + str(selected_community) + ' k=' + str(k) + '.json',
                    #     gml_number, seed_set)
                    # print('p=' + str(p) + ' attr=' + str(selected_community) + ' k=' + str(k) + ' gml_num=' + str(
                    #     gml_number))

def main_DC_networkgml():
    ps = [0.1]
    RRs_num = 253858
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    ks = [10]
    file_path = "data/networks_gml"
    selected_communities = ["ethnicity"]
    concave_function_types = [2]
    function_parameters_dic = {
        2: [40],
    }
    for p in ps:
        for selected_community in selected_communities:
            for k in ks:
                for gml_number in range(1):
                    edge_list, G = load_graph_based_on_igraph(file_path + '/graph_spa_500_' + str(gml_number) + '.gml',
                                                              selected_community)
                    communities = G.Communities
                    cff = IMM(G, k, approx_rate, err_pr, graph_type)
                    cff.set_edge_prob(p)
                    RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                                 graph_type)
                    for concave_function_type in concave_function_types:
                        for function_parameter in function_parameters_dic[concave_function_type]:
                            seed_set = cff.degree_concave_nodeselection(RRs_set, concave_function_type,
                                                                        function_parameter, 5 / 34)
                            with open('zjx_results/gml_result/DC_p=0.1_k=10_RRs_253858.txt', 'w') as f:
                                f.write(str(seed_set))
                            print('DC:', seed_set)

def main_IMM_karate():
    ps = [0.1]
    RRs_num = 100000
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    ks = [5]
    file_path = "data/karate/karate-edge.txt"
    community_file = 'data/karate/karate-community.txt'
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            seed_set = cff.NodeSelection(RRs_set)
            with open('IMM_result/p=0.1 k=5.txt', 'w') as f:
                f.write(str(seed_set))


def main_CFF_karate():
    ps = [0.1]
    RRs_num = 100000
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    concave_function_types = [2]
    function_parameters_dic = {
        2: [40],
    }
    ks = [5]
    file_path = "data/karate/karate-edge.txt"
    community_file = 'data/karate/karate-community.txt'
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            for concave_function_type in concave_function_types:
                for function_parameter in function_parameters_dic[concave_function_type]:
                    seed_set = cff.CELF_community_fair_NodeSelection(RRs_set, concave_function_type,
                                                                     function_parameter)
                    # with open('CFF_result/p=0.1 k=5.txt', 'w') as f:
                    #     f.write(str(seed_set))
                    print('CFF:', seed_set)

def main_DC_karate():
    ps = [0.1]
    RRs_num = 100000
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    ks = [5]
    file_path = "data/karate/karate-edge.txt"
    community_file = 'data/karate/karate-community.txt'
    concave_function_types = [2]
    function_parameters_dic = {
        2: [40],
    }
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            for concave_function_type in concave_function_types:
                for function_parameter in function_parameters_dic[concave_function_type]:
                    seed_set = cff.degree_concave_nodeselection(RRs_set, concave_function_type,
                                                                function_parameter, 5 / 34)
                    # with open('DC_result/p=0.1 k=5.txt', 'w') as f:
                    #     f.write(str(seed_set))
                    print('DC:', seed_set)


def save_result(path, gml_num, seed_set):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('[')
    with open(path, 'a') as f:
        result = {'gml_num': str(gml_num),
                  'seed_set': str(seed_set)}
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
        if gml_num < 23:
            f.write(',')
        f.write('\r\n')
        if gml_num == 23:
            f.write(']')


def temp_cal_PoF_RoF():
    file_path = "data/karate/karate-edge.txt"
    community_file = 'data/karate/karate-community.txt'
    round_num = 15
    edge_list, G = load_graph(file_path, community_file)
    cff = IMM(G, 5, 0.1, 0.5, 0)
    cff.set_edge_prob(0.1)
    IMM_seed_set = [34, 1, 33, 2, 25]
    # CFF_seed_set = [1, 34, 17, 32, 3]
    CFF_seed_set = [1, 34, 4, 33, 9]
    CFF_ap, IMM_ap = np.zeros(35), np.zeros(35)
    CFF_influence, IMM_influence = 0, 0
    for i in range(round_num):
        temp_CFF_ap, temp_CFF_influence = cff.multi_process_MC_InfEst(edge_list, 2, CFF_seed_set, 10000)
        temp_IMM_ap, temp_IMM_influence = cff.multi_process_MC_InfEst(edge_list, 2, IMM_seed_set, 10000)
        CFF_influence += temp_CFF_influence / round_num
        IMM_influence += temp_IMM_influence / round_num
        CFF_ap += temp_CFF_ap / round_num
        IMM_ap += temp_IMM_ap / round_num

    CFF_utilities = [0] * (cff.graph.Communities.size + 2)
    IMM_utilities = [0] * (cff.graph.Communities.size + 2)
    for cID in cff.graph.Communities.communities:
        for node in cff.graph.Communities.communities[cID].community:
            CFF_utilities[cID] += CFF_ap[node] / cff.graph.Communities.communities[cID].size
            IMM_utilities[cID] += IMM_ap[node] / cff.graph.Communities.communities[cID].size

    RoF = (np.var(IMM_utilities[2:]) - np.var(CFF_utilities[2:])) / (
            np.var(IMM_utilities[2:]) + np.var(CFF_utilities[2:]))
    PoF = (IMM_influence - CFF_influence) / IMM_influence
    Fscore = 5 * (1 + RoF) / 2 * (1 - PoF) / (2 * (1 + RoF) + (1 - PoF))
    print('RoF:', RoF)
    print('PoF:', PoF)


def temp_draw_karate():
    file_path = "data/karate/karate-edge-source.txt"
    community_file = 'data/karate/karate-community-source.txt'
    edge_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line_split = line.split()
            line_split = list(map(int, line_split))
            edge_list.append(line_split)

    g = nx.Graph()
    n = len(edge_list)
    for i in range(n):
        g.add_edge(edge_list[i][0], edge_list[i][1])

    with open(community_file, 'r') as f:
        for line in f:
            line_split = line.split()
            line_split = list(map(int, line_split))
            g.add_node(line_split[0], label1=str(line_split[0] + 1), community=str(line_split[1]))

    nx.write_gml(g, 'data/karate/karate.gml')


def temp_get_runtime():
    ps = [0.1]
    RRs_num = 100000
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 1
    concave_function_types = [1, 3, 4, 2, 0]
    function_parameters_dic = {
        1: [0.1],
        3: [2],
        4: [0.1],
        2: [40],
        0: [0.1]
    }
    # ks = [5, 10, 15, 20]
    ks = [10]
    file_path = "data/networks_gml"
    selected_communities = ["age"]
    for p in ps:
        for selected_community in selected_communities:
            for k in ks:
                for gml_number in range(1):
                    edge_list, G = load_graph_based_on_igraph(file_path + '/graph_spa_500_' + str(gml_number) + '.gml',
                                                              selected_community)
                    communities = G.Communities
                    cff = IMM(G, k, approx_rate, err_pr, graph_type)
                    cff.set_edge_prob(p)
                    RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                                 graph_type)
                    for concave_function_type in concave_function_types:
                        for function_parameter in function_parameters_dic[concave_function_type]:
                            start_time = time.time()
                            seed_set = cff.CELF_community_fair_NodeSelection(RRs_set, concave_function_type,
                                                                             function_parameter)
                            print(concave_function_type, function_parameter, time.time() - start_time, seed_set)

def main_CFF_email():
    ps = [0.01]
    RRs_num = 294938
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    concave_function_types = [2]
    function_parameters_dic = {
        2: [40],
    }
    ks = [10]
    file_path = 'data/email/email-Eu-core.txt'
    community_file = 'data/email/email-Eu-core-department-labels.txt'
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            for concave_function_type in concave_function_types:
                for function_parameter in function_parameters_dic[concave_function_type]:
                    seed_set = cff.CELF_community_fair_NodeSelection(RRs_set, concave_function_type,
                                                                     function_parameter)
                    with open('zjx_results/email_result/CFF_p=0.01_k=10.txt', 'w') as f:
                        f.write(str(seed_set))
                    print('CFF:', seed_set)

def main_DC_email():
    ps = [0.01]
    RRs_num = 294938
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    ks = [10]
    file_path = 'data/email/email-Eu-core.txt'
    community_file = 'data/email/email-Eu-core-department-labels.txt'
    concave_function_types = [2]
    function_parameters_dic = {
        2: [40],
    }
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            for concave_function_type in concave_function_types:
                for function_parameter in function_parameters_dic[concave_function_type]:
                    seed_set = cff.degree_concave_nodeselection(RRs_set, concave_function_type,
                                                                function_parameter, 5 / 34)
                    with open('zjx_results/email_result/DC_p=0.01_k=10.txt', 'w') as f:
                        f.write(str(seed_set))
                    print('DC:', seed_set)

def main_IMM_email():
    ps = [0.01]
    RRs_num = 294938
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    ks = [10]
    file_path = 'data/email/email-Eu-core.txt'
    community_file = 'data/email/email-Eu-core-department-labels.txt'
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            seed_set = cff.NodeSelection(RRs_set)
            with open('zjx_results/email_result/IMM_p=0.01_k=10.txt', 'w') as f:
                f.write(str(seed_set))
            print('IMM:', seed_set)

def main_CFF_UVM():
    ps = [0.01]
    RRs_num = 227818
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    concave_function_types = [2]
    function_parameters_dic = {
        2: [40],
    }
    ks = [10]
    file_path = 'data/UVM/uvm-edges.txt'
    community_file = 'data/UVM/uvm-labels-zjx.txt'
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            for concave_function_type in concave_function_types:
                for function_parameter in function_parameters_dic[concave_function_type]:
                    seed_set = cff.CELF_community_fair_NodeSelection(RRs_set, concave_function_type,
                                                                     function_parameter)
                    with open('zjx_results/UVM_result/CFF_p=0.01_k=10_RRS=227818.txt', 'w') as f:
                        f.write(str(seed_set))
                    print('CFF:', seed_set)

def main_IMM_UVM():
    ps = [0.01]
    RRs_num = 227818
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    ks = [10]
    file_path = 'data/UVM/uvm-edges.txt'
    community_file = 'data/UVM/uvm-labels-zjx.txt'
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            seed_set = cff.NodeSelection(RRs_set)
            with open('zjx_results/UVM_result/IMM_p=0.01_k=10_RRS=227818.txt', 'w') as f:
                f.write(str(seed_set))
            print('IMM:', seed_set)

def main_DC_UVM():
    ps = [0.01]
    RRs_num = 227818
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    ks = [10]
    file_path = 'data/UVM/uvm-edges.txt'
    community_file = 'data/UVM/uvm-labels-zjx.txt'
    concave_function_types = [2]
    function_parameters_dic = {
        2: [40],
    }
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            for concave_function_type in concave_function_types:
                for function_parameter in function_parameters_dic[concave_function_type]:
                    seed_set = cff.degree_concave_nodeselection(RRs_set, concave_function_type,
                                                                function_parameter, 5 / 34)
                    with open('zjx_results/UVM_result/DC_p=0.01_k=10_RRS=227818.txt', 'w') as f:
                        f.write(str(seed_set))
                    print('DC:', seed_set)

def main_CFF_youtube():
    ps = [0.1]
    RRs_num = 47463
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    concave_function_types = [2]
    function_parameters_dic = {
        2: [40],
    }
    ks = [10]
    file_path = 'data/youtube/youtube-edges.txt'
    community_file = 'data/youtube/youtube-labels.txt'
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            for concave_function_type in concave_function_types:
                for function_parameter in function_parameters_dic[concave_function_type]:
                    seed_set = cff.CELF_community_fair_NodeSelection(RRs_set, concave_function_type,
                                                                     function_parameter)
                    with open('zjx_results/youtube_result/CFF_p=0.1_k=10_RRS=47463.txt', 'w') as f:
                        f.write(str(seed_set))
                    print('CFF:', seed_set)

def main_IMM_youtube():
    ps = [0.1]
    RRs_num = 47463
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    ks = [10]
    file_path = 'data/youtube/youtube-edges.txt'
    community_file = 'data/youtube/youtube-labels.txt'
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)

            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            seed_set = cff.NodeSelection(RRs_set)
            with open('zjx_results/youtube_result/IMM_p=0.1_k=10_RRS=47463.txt', 'w') as f:
                f.write(str(seed_set))
            print('IMM:', seed_set)

def main_DC_youtube():
    ps = [0.1]
    RRs_num = 47463
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 2
    graph_type = 0
    ks = [10]
    file_path = 'data/youtube/youtube-edges.txt'
    community_file = 'data/youtube/youtube-labels.txt'
    concave_function_types = [2]
    function_parameters_dic = {
        2: [40],
    }
    for p in ps:
        for k in ks:
            edge_list, G = load_graph(file_path, community_file)
            communities = G.Communities
            cff = IMM(G, k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(p)
            RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p,
                                         graph_type)
            for concave_function_type in concave_function_types:
                for function_parameter in function_parameters_dic[concave_function_type]:
                    seed_set = cff.degree_concave_nodeselection(RRs_set, concave_function_type,
                                                                function_parameter, 5 / 34)
                    with open('zjx_results/youtube_result/DC_p=0.1_k=10_RRS=47463.txt', 'w') as f:
                        f.write(str(seed_set))
                    print('DC:', seed_set)

def main_gml_emo():
    k = 10
    ps = [0.1]
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 8
    RRs_num = 253858
    graph_type = 1
    attribute = 'department'
    file = 'data/gml/gml0-edges.txt'
    # community_file = 'data/gml/gml0-department-labels.txt'
    community_file = 'data/gml/gml0-department-labels(start0).txt'
    c_ps = [0.75]
    m_ps = [0.2]
    fc_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # set to 0.2
    fm_ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # set to 0.7
    N = 100
    T = 50
    for p in ps:
        edge_list, G = load_graph(file, community_file)
        communities = G.Communities
        for key in communities.communities:
            community = communities.communities[key]
        cff = IMM(G, k, approx_rate, err_pr, graph_type)
        cff.set_edge_prob(p)

        RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core,
                                     communities, p,
                                     graph_type)

        result_folder = 'zjx_results/gml_emo/'
        # seed_sets_path = 'seed_set/seed_DD.txt'
        # seed_sets = load_seed_sets(seed_sets_path)

        seed_De = load_seed_sets('zjx_results/gml_result/De.txt')
        seed_DD = load_seed_sets('zjx_results/gml_result/DD.txt')
        seed_IMM = load_seed_sets('zjx_results/gml_result/IMM_p=0.1_k=10_RRs_253858.txt')
        seed_DC = load_seed_sets('zjx_results/gml_result/DC_p=0.1_k=10_RRs_253858.txt')
        seed_CFF = load_seed_sets('zjx_results/gml_result/CFF_p=0.1_k=10_RRs_253858.txt')
        seed_WF = load_seed_sets('zjx_results/gml_result/WF.txt')

        # seed_combine = [seed_De[0]] + [seed_DD[0]] + [seed_IMM[0]] + [seed_DC[0]] + [seed_CFF[0]] + [seed_WF[0]]
        # seed_combine = [seed_De[0]] + [seed_DD[0]] + [seed_IMM[0]] + [seed_CFF[0]] + [seed_WF[0]]
        seed_combine = []

        for c_p in c_ps:
            for m_p in m_ps:
                for fc_ratio in fc_ratios:
                    for fm_p in fm_ps:
                        result_file = 'c_p=' + str(c_p) + ' m_p=' + str(m_p) + ' fc_ratio=' + str(
                            fc_ratio) + ' fm_p=' + str(fm_p) + ' N=' + str(N) + ' T=' + str(T) + '.json'
                        mo = emo(cff, c_p, m_p, fc_ratio, fm_p, N, 2, k, T, [1, -1], result_folder + result_file,
                                 'gml_result', attribute, edge_list, cpu_core, 'RFMFIM')
                        mo.fair_nsga2(RRs_set, seed_combine)
                        # mo.fair_nsga2(RRs_set, seed_DD)

def main_uvm_emo():
    k = 10
    ps = [0.01]
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 8
    RRs_num = 227818
    graph_type = 1
    attribute = 'department'
    file = 'data/uvm/uvm-edges.txt'
    # community_file = 'data/gml/gml0-department-labels.txt'
    community_file = 'data/uvm/uvm-labels-zjx.txt'
    c_ps = [0.75]
    m_ps = [0.2]
    fc_ratios = [0]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # set to 0.2
    fm_ps = [0]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # set to 0.7
    N = 30
    T = 30
    for p in ps:
        edge_list, G = load_graph(file, community_file)
        communities = G.Communities
        for key in communities.communities:
            community = communities.communities[key]
        cff = IMM(G, k, approx_rate, err_pr, graph_type)
        cff.set_edge_prob(p)

        RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core,
                                     communities, p,
                                     graph_type)

        result_folder = 'zjx_results/uvm_emo/'
        # seed_sets_path = 'seed_set/seed_DD.txt'
        # seed_sets = load_seed_sets(seed_sets_path)

        seed_De = load_seed_sets('zjx_results/UVM_result/De.txt')
        seed_DD = load_seed_sets('zjx_results/UVM_result/DD.txt')
        seed_IMM = load_seed_sets('zjx_results/UVM_result/IMM_p=0.01_k=10_RRS=227818.txt')
        # seed_DC = load_seed_sets('zjx_results/UVM_result/DC_p=0.01_k=10_RRS=227818.txt')
        seed_CFF = load_seed_sets('zjx_results/UVM_result/CFF_p=0.01_k=10_RRS=227818.txt')
        seed_WF = load_seed_sets('zjx_results/UVM_result/WF.txt')

        # seed_combine = [seed_De[0]] + [seed_DD[0]] + [seed_IMM[0]] + [seed_DC[0]] + [seed_CFF[0]] + [seed_WF[0]]
        seed_combine = [seed_De[0]] + [seed_DD[0]] + [seed_IMM[0]] + [seed_CFF[0]] + [seed_WF[0]]
        # seed_combine = []

        for c_p in c_ps:
            for m_p in m_ps:
                for fc_ratio in fc_ratios:
                    for fm_p in fm_ps:
                        result_file = 'c_p=' + str(c_p) + ' m_p=' + str(m_p) + ' fc_ratio=' + str(
                            fc_ratio) + ' fm_p=' + str(fm_p) + ' N=' + str(N) + ' T=' + str(T) + '.json'
                        mo = emo(cff, c_p, m_p, fc_ratio, fm_p, N, 2, k, T, [1, -1], result_folder + result_file,
                                 'uvm_result', attribute, edge_list, cpu_core, 'RFMFIM')
                        mo.fair_nsga2(RRs_set, seed_combine)
                        # mo.fair_nsga2(RRs_set, seed_DD)

def main_youtube_emo():
    k = 10
    ps = [0.05]
    approx_rate = 0.1
    err_pr = 0.5
    cpu_core = 8
    RRs_num = 11235*8
    graph_type = 1
    attribute = 'department'
    file = 'data/youtube/youtubeNEW-edges.txt'
    # community_file = 'data/gml/gml0-department-labels.txt'
    community_file = 'data/youtube/youtubeNEW-labels.txt'
    c_ps = [0.75]
    m_ps = [0.2]
    fc_ratios = [0]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # set to 0.2
    fm_ps = [0]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # set to 0.7
    N = 30
    T = 30
    for p in ps:
        edge_list, G = load_graph(file, community_file)
        communities = G.Communities
        for key in communities.communities:
            community = communities.communities[key]
        cff = IMM(G, k, approx_rate, err_pr, graph_type)
        cff.set_edge_prob(p)

        RRs_set = multi_generate_RRs(edge_list, k, approx_rate, err_pr, RRs_num, cpu_core, communities, p, graph_type)

        result_folder = 'zjx_results/youtube_emo/'
        # seed_sets_path = 'seed_set/seed_DD.txt'
        # seed_sets = load_seed_sets(seed_sets_path)

        seed_De = load_seed_sets('zjx_results/youtube_result/De.txt')
        seed_DD = load_seed_sets('zjx_results/youtube_result/DD.txt')
        seed_IMM = load_seed_sets('zjx_results/youtube_result/IMM.txt')
        seed_DC = load_seed_sets('zjx_results/youtube_result/DC.txt')
        seed_CFF = load_seed_sets('zjx_results/youtube_result/CFF.txt')
        seed_WF = load_seed_sets('zjx_results/youtube_result/WF.txt')

        # seed_combine = [seed_De[0]] + [seed_DD[0]] + [seed_IMM[0]] + [seed_DC[0]] + [seed_CFF[0]] + [seed_WF[0]]
        seed_combine = [seed_De[0]] + [seed_DD[0]] + [seed_IMM[0]] + [seed_CFF[0]] + [seed_WF[0]]
        # seed_combine = []

        for c_p in c_ps:
            for m_p in m_ps:
                for fc_ratio in fc_ratios:
                    for fm_p in fm_ps:
                        result_file = 'c_p=' + str(c_p) + ' m_p=' + str(m_p) + ' fc_ratio=' + str(
                            fc_ratio) + ' fm_p=' + str(fm_p) + ' N=' + str(N) + ' T=' + str(T) + '.json'
                        mo = emo(cff, c_p, m_p, fc_ratio, fm_p, N, 2, k, T, [1, -1], result_folder + result_file,
                                 'youtube_result', attribute, edge_list, cpu_core, 'RFMFIM')
                        mo.fair_nsga2(RRs_set, seed_combine)
                        # mo.fair_nsga2(RRs_set, seed_DD)

if __name__ == '__main__':

    # main_CFF_youtube()
    # main_IMM_youtube()
    # main_DC_youtube()

    # main_CFF_UVM()
    # main_IMM_UVM()
    # main_DC_UVM()


    # main_CFF_networksgml()
    # main_IMM_networkgml()
    # main_DC_networkgml()

    # main_CFF_email()
    # main_DC_email()
    # main_IMM_email()

    main_EMO_email()
    # main_gml_emo()
    # main_uvm_emo()
    # main_youtube_emo()

    # json_result_path = 'zjx_results/gml_emo/'  # Path to your JSON files
    # DrawGraph(json_result_path)  # Create an instance of DrawGraph
    # main_CFF_networksgml()
    # main_IMM_networkgml()

    # main_CFF_karate()
    # main_IMM_karate()
    # main_DC_karate()

    # temp_cal_PoF_RoF()
    # temp_draw_karate()
    # temp_get_runtime()

