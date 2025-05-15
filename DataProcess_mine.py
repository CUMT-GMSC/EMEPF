import json
import os
from collections import Counter

from igraph import Graph

import numpy as np

from CFF_model import IMM
from LoadData import load_graph_based_on_igraph


class Result:
    def __init__(self, CFF_result_path, IMM_result_path, date):
        self.CFF_result_path = CFF_result_path
        self.IMM_result_path = IMM_result_path
        self.date = date
        self.load_result()
        self.cal_PoF()

    def load_result(self):
        self.p, self.attr, self.k = self.CFF_result_path.split()
        self.p = float(self.p[26:])
        self.attr = self.attr[5:]
        self.k = int(self.k[2:-5])

        with open(self.CFF_result_path, 'r') as f:
            self.CFF_result_data = json.load(f)

        with open(self.IMM_result_path, 'r') as f:
            self.IMM_result_data = json.load(f)

    def cal_PoF(self):
        approx_rate = 0.1
        err_pr = 0.5
        graph_type = 1
        round_num = 15
        file_path = "data/networks_gml"
        RoF_list = []
        PoF_list = []
        Fscore_list = []
        for gml_num in range(24):
            edge_list, G = load_graph_based_on_igraph(file_path + '/graph_spa_500_' + str(gml_num) + '.gml', self.attr)
            cff = IMM(G, self.k, approx_rate, err_pr, graph_type)
            cff.set_edge_prob(self.p)

            str_list = self.CFF_result_data[gml_num]['seed_set']
            str_list = str_list.strip('[]')
            str_list = str_list.split(',')
            CFF_seed_set = [int(num) for num in str_list]

            str_list = self.IMM_result_data[gml_num]['seed_set']
            str_list = str_list.strip('[]')
            str_list = str_list.split(',')
            IMM_seed_set = [int(num) for num in str_list]

            CFF_ap, IMM_ap = np.zeros(501), np.zeros(501)
            CFF_influence, IMM_influence = 0, 0
            for i in range(round_num):
                temp_CFF_ap, temp_CFF_influence = cff.multi_process_MC_InfEst(edge_list, 2, CFF_seed_set, 10000)
                temp_IMM_ap, temp_IMM_influence = cff.multi_process_MC_InfEst(edge_list, 2, IMM_seed_set, 10000)
                CFF_influence += temp_CFF_influence / round_num
                IMM_influence += temp_IMM_influence / round_num
                CFF_ap += temp_CFF_ap / round_num
                IMM_ap += temp_IMM_ap / round_num

            CFF_utilities = [0] * (cff.graph.Communities.size + 1)
            IMM_utilities = [0] * (cff.graph.Communities.size + 1)
            for cID in cff.graph.Communities.communities:
                for node in cff.graph.Communities.communities[cID].community:
                    CFF_utilities[cID] += CFF_ap[node] / cff.graph.Communities.communities[cID].size
                    IMM_utilities[cID] += IMM_ap[node] / cff.graph.Communities.communities[cID].size

            RoF = (np.var(IMM_utilities[1:]) - np.var(CFF_utilities[1:])) / (
                    np.var(IMM_utilities[1:]) + np.var(CFF_utilities[1:]))
            PoF = (IMM_influence - CFF_influence) / IMM_influence
            Fscore = 5 * (1 + RoF) / 2 * (1 - PoF) / (2 * (1 + RoF) + (1 - PoF))

            RoF_list.append(RoF)
            PoF_list.append(PoF)
            Fscore_list.append(Fscore)

            print(gml_num, 'RoF=', RoF, 'PoF=', PoF, 'Fscore=', Fscore)
            with open('result2/' + str(self.date) + '/RoF_PoF_Fscore_result/attr=' + str(self.attr) + ' k=' + str(
                    self.k) + '.txt', 'a') as f:
                if gml_num == 0:
                    f.write('gml_num RoF PoF Fscore\n')
                f.write(str(gml_num) + ' ' + str(RoF) + ' ' + str(PoF) + ' ' + str(Fscore))
                f.write('\n')
                if gml_num == 23:
                    f.write('mean ' + str(np.mean(RoF_list)) + ' ' + str(np.mean(PoF_list)) + ' ' + str(
                        np.mean(Fscore_list)))


def cal_var(file_path):
    RoF = [float] * 24
    PoF = [float] * 24
    Fscore = [float] * 24
    with open(file_path, 'r') as f:
        f.readline()
        for i in range(24):
            line = f.readline().split()
            RoF[i] = float(line[1])
            PoF[i] = float(line[2])
            Fscore[i] = float(line[3])

    with open(file_path, 'a') as f:
        f.write('\n')
        f.write('var ' + str(np.var(RoF)) + ' ' + str(np.var(PoF)) + ' ' + str(np.var(Fscore)))
        f.write('\n')
        f.write('std ' + str(np.std(RoF)) + ' ' + str(np.std(PoF)) + ' ' + str(np.std(Fscore)))


def cal_group_size1():
    file_path = "data/networks_gml"
    ethnicity = []
    age = []
    gender = []

    for gml_number in range(24):
        g = Graph.Read_GML(file_path + '/graph_spa_500_' + str(gml_number) + '.gml')
        ethnicity += g.vs['ethnicity']
        age += g.vs['age']
        gender += g.vs['gender']

    ethnicity_groupsize = Counter(ethnicity)
    age_groupsize = Counter(age)
    gender_groupsize = Counter(gender)

    ethnicity_groupsize = {key: value / ethnicity_groupsize.total() for key, value in ethnicity_groupsize.items()}
    age_groupsize = {key: value / age_groupsize.total() for key, value in age_groupsize.items()}
    gender_groupsize = {key: value / gender_groupsize.total() for key, value in gender_groupsize.items()}

    print(ethnicity_groupsize, age_groupsize, gender_groupsize)

    
    return

def cal_group_size():
    file_path = "data/networks_gml"
    ethnicity = [Counter()] * 24
    age = [Counter()] * 24
    gender = [Counter()] * 24

    for gml_number in range(24):
        g = Graph.Read_GML(file_path + '/graph_spa_500_' + str(gml_number) + '.gml')
        ethnicity[gml_number] = Counter(g.vs['ethnicity'])
        age[gml_number] = Counter(g.vs['age'])
        gender[gml_number] = Counter(g.vs['gender'])

    print('*********************ethnicity_group_size_var*********************')
    print('key', 'var', 'size')
    for key in ethnicity[0].keys():
        temp = [ethnicity[gml_number][key] for gml_number in range(24)]
        ethnicity_var = np.var(temp)
        print(key, ethnicity_var, temp)
    print('*********************age_group_size_var*********************')
    print('key', 'var', 'size')
    for key in age[0].keys():
        temp = [age[gml_number][key] for gml_number in range(24)]
        age_var = np.var(temp)
        print(key, age_var, temp)
    print('*********************gender_group_size_var*********************')
    print('key', 'var', 'size')
    for key in gender[0].keys():
        temp = [gender[gml_number][key] for gml_number in range(24)]
        gender_var = np.var(temp)
        print(key, gender_var, temp)

    print('')
    print('*********************group_size*********************')
    ethnicity_sum = Counter()
    age_sum = Counter()
    gender_sum = Counter()

    for gml_number in range(24):
        ethnicity_sum += ethnicity[gml_number]
        age_sum += age[gml_number]
        gender_sum += gender[gml_number]

    ethnicity_sum = {key: value/ethnicity_sum.total() for key, value in ethnicity_sum.items()}
    age_sum = {key: value/age_sum.total() for key, value in age_sum.items()}
    gender_sum = {key: value/gender_sum.total() for key, value in gender_sum.items()}

    print('ethnicity:', ethnicity_sum)
    print('age:', age_sum)
    print('gender:', gender_sum)

if __name__ == '__main__':
    date = '6.26'
    # name_list = ['p=0.1 attr=age k=5.json', 'p=0.1 attr=age k=10.json', 'p=0.1 attr=age k=15.json',
    #              'p=0.1 attr=age k=20.json',
    #              'p=0.1 attr=ethnicity k=5.json', 'p=0.1 attr=ethnicity k=10.json', 'p=0.1 attr=ethnicity k=15.json',
    #              'p=0.1 attr=ethnicity k=20.json',
    #              'p=0.1 attr=gender k=5.json', 'p=0.1 attr=gender k=10.json', 'p=0.1 attr=gender k=15.json',
    #              'p=0.1 attr=gender k=20.json']
    name_list = ['attr=age k=25.txt',
                 'attr=ethnicity k=25.txt',
                 'attr=gender k=25.txt']
    # for name in name_list:
    # Result('result2/' + date + '/CFF_result/' + name, 'result2/' + date + '/IMM_result/' + name, date)
    # print(name + ' finished')

    # cal_var('result2/' + date + '/RoF_PoF_Fscore_result/' + name)

    cal_group_size()
