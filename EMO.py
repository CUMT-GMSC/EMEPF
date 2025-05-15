import json
import math
import pickle
import time

import Cal_HV
from graph import *
import networkx as nx
import numpy as np
import sys
import random
from CFF_model import IMM
import copy
from Measure import resultData
from beautifulJson import NoIndent,MyEncoder

class emo(object):
    def __init__(self,imm:IMM,c_p,m_p,fc_ratio,fm_p,N,M,k,T,better,file, graphName,attribute,edge_list,cpu_core,alg_name ):
        self.imm = imm
        self.graph = self.imm.graph   # 图，Graph类
        self.c_p = c_p    # crossover概率
        self.m_p = m_p    # mutation概率
        self.fc_ratio = fc_ratio  # f0fronts 占 crossover的比例  [0,1]
        self.fm_p = fm_p  # 发生fair mutation的概率        [0,1]
        self.N = N        # 种群数量
        self.M = M        # 目标函数数量        2
        self.k = self.imm.k        # 种子集规模
        self.T = T        # 演化时间
        self.better = better     # list 描述目标函数是越大越好还是越小越好，1表示越大越好，-1表示越小越好    [1, -1]
        self.population = np.empty((N,k),dtype=int)
        self.file = file       # 结果文件夹
        self.graphName = graphName
        self.attribute = attribute
        self.edge_list = edge_list
        self.cpu_core = cpu_core    # 进程数量
        self.alg_name = alg_name    #

    def dominate(self, objectives1, objectives2):
        dom1, dom2 = 0, 0
        l = len(objectives1)
        for o1, o2, b in zip(objectives1, objectives2, self.better):
            temp = o1 - o2
            if temp == 0:
                dom1 += 1
                dom2 += 1
            elif temp * b > 0:
                dom1 += 1
            elif temp * b < 0:
                dom2 += 1
        if dom1 == l and dom2 < l:
            return 1
        elif dom2 == l and dom1 < l:
            return -1
        else:
            return 0

    def fast_non_dominated_sort(self, objs: np.ndarray):
        n = np.zeros(objs.shape[0], dtype=int)  # n[p]表示dominate p的总数量
        S = {}  # S[p]表示p dominate的 solutions的集合，数据类型是list
        rank = np.zeros(objs.shape[0], dtype=int)  # rank[p]表示p的front层
        F = {0: []}  # F[i]表示第i front的个体集
        for p_number in range(objs.shape[0]):
            S[p_number] = []
            for q_number in range(objs.shape[0]):
                if p_number != q_number:
                    p = objs[p_number]
                    q = objs[q_number]
                    domination = self.dominate(p, q)
                    if domination == 1:
                        S[p_number].append(q_number)
                    elif domination == -1:
                        n[p_number] += 1
            if n[p_number] == 0:
                F[0].append(p_number)
        i = 0
        while F[i]:  # 如果F[i]不为空
            Q = []
            for p_number in F[i]:
                for q_number in S[p_number]:
                    n[q_number] -= 1
                    if n[q_number] == 0:
                        rank[q_number] = i + 1
                        Q.append(q_number)
            i += 1
            F[i] = Q
        del F[i]
        return F, rank

    def crowding_distance(self, I: list, objectives: np.ndarray):
        l = len(I)
        distance = {}
        for item in I:
            distance[item] = 0
        I_objectives = np.empty([0, self.M])
        for p_number in I:
            objective = objectives[p_number]
            I_objectives = np.concatenate((I_objectives, [objective]))

        for i in range(self.M):
            sorted_I_index = np.argsort(I_objectives[:, i])
            distance[I[sorted_I_index[0]]] = sys.maxsize
            distance[I[sorted_I_index[l - 1]]] = sys.maxsize
            for j in range(1, l - 1):
                if distance[I[sorted_I_index[j]]] != sys.maxsize:
                    distance[I[sorted_I_index[j]]] += (I_objectives[sorted_I_index[j + 1]][i] -
                                                       I_objectives[sorted_I_index[j - 1]][i]) / (
                                                                  I_objectives[sorted_I_index[l - 1]][i] -
                                                                  I_objectives[sorted_I_index[0]][i])
        return distance

    def crowded_comparison_operator(self, r1, d1, r2, d2):
        if r1 == r2:
            if d1 == d2:
                return 0
            elif d1 > d2:
                return 1
            else:
                return -1
        elif r1 < r2:
            return 1
        else:
            return -1

    def tournament_selection(self, rank: np.ndarray, distance: dict):
        a = random.randint(0, self.N - 1)
        b = random.randint(0, self.N - 1)
        c = random.randint(0, self.N - 1)
        d = random.randint(0, self.N - 1)
        r1 = self.crowded_comparison_operator(rank[a], distance[a], rank[b], distance[b])
        r2 = self.crowded_comparison_operator(rank[c], distance[c], rank[d], distance[d])
        if r1 < 0:
            a = b
        if r2 < 0:
            c = d
        return [a, c]

    def crossover(self, parents):
        p1 = self.population[parents[0]].copy()
        p2 = self.population[parents[1]].copy()
        l = len(p1)
        for i in range(l):
            if random.random() <= self.c_p and p1[i] not in p2 and p2[i] not in p1:
                temp = p1[i]
                p1[i] = p2[i]
                p2[i] = temp
        return [p1, p2]

    def mutation(self, p):
        for i in range(self.k):
            if random.random() <= self.m_p:
                p[i] = random.choice(list(set([i for i in range(1, self.graph.nodes_number+1)]).difference(set(p))))
        return p

    @staticmethod
    def one_multinomial_exp(dic: dict):
        p = [value for value in dic.values()]
        a = np.random.multinomial(1, p, 1)
        index = np.nonzero(a)[1][0]
        key = list(dic.keys())[index]
        return key

    def fair_mutation(self, population, B, nodeID_RRs_set_dict):
        # Fi = max(0, (\bar{u} - ui))
        # Pi = Fi / sum(Fi)
        influcenced_RRs_set = self.get_influenced_RRs_set(population, nodeID_RRs_set_dict)
        ap, utilities, influence = self.influence(population, influcenced_RRs_set)
        avg_utilities = np.mean(utilities)
        Fi_dict = {cID: max(0, avg_utilities - utilities[cID]) for cID in self.graph.Communities.communities.keys()}
        Fi_sum = sum(Fi_dict.values())
        if Fi_sum == 0:
            Fi_sum = 1e-10
        Pi_dict = {cID: Fi_dict[cID] / Fi_sum for cID in self.graph.Communities.communities.keys()}
        for i in range(self.k):
            if random.random() <= self.m_p:
                cID = self.one_multinomial_exp(Pi_dict)
                if len(list(set(B[cID]).difference(set(population)))) > 0:
                    mutation_result = random.choice(B[cID])
                    while mutation_result in population:
                        mutation_result = random.choice(B[cID])
                    population[i] = mutation_result
                else:
                    population[i] = random.choice(list(set([i for i in range(1, self.graph.nodes_number+1)]).difference(set(population))))

        return population

    def init_population_prior_seed_set(self, seed_sets):
        node_list = [i for i in range(1, self.graph.nodes_number + 1)]
        p_number = 0
        if len(seed_sets) > 0:
            for seed_set in seed_sets:
                p = np.array(seed_set)
                self.population[p_number] = p
                p_number += 1
        while p_number < self.N:
            p = random.sample(list(node_list), self.k)
            p = np.array(p)
            self.population[p_number] = p
            p_number += 1

    def influence(self, seed_set: list, RRs_set):
        ap = self.imm.RRs2ap2(seed_set, RRs_set, self.N_RRs_set)
        utilities = self.imm.ap2utility(ap)
        influence = sum(ap)
        return ap, utilities, influence

    def get_nodeID_RRs_set_dict(self, RRs_set):
        nodeID_RRs_set_dict = {nodeID: [] for nodeID in range(1, self.graph.nodes_number + 1)}
        for RRs in RRs_set:
            for nodeID in RRs[1:]:
                nodeID_RRs_set_dict[nodeID].append(RRs)
        return nodeID_RRs_set_dict

    @staticmethod
    def get_influenced_RRs_set(seed_set, nodeID_RRs_set_dict):
        influcenced_RRs_set = []
        for nodeID in seed_set:
            for RRs in nodeID_RRs_set_dict[nodeID]:
                if RRs not in influcenced_RRs_set:
                    influcenced_RRs_set.append(RRs)
        return influcenced_RRs_set

    def gen_B(self, node_ID_RRs_set_dict):
        # for each seed in community[cID], add all nodes_influence_seed to B[cID]
        # B = {cID: seed_set: list ...} not unique
        B = {cID: [] for cID in self.graph.Communities.communities.keys()}
        print(self.graph.Communities.communities.keys(), len(self.graph.Communities.communities.keys()))
        for cID in self.graph.Communities.communities.keys():
            for nodeID in self.graph.Communities.communities[cID].community:
                for RRs in node_ID_RRs_set_dict[nodeID]:
                    B[cID] += RRs[1:]
            print(len(B[cID]), cID)
        return B

    @staticmethod
    def x_var(utilities):
        # 对 utilities 归一化
        sum_utilities = sum(utilities)
        x_utilities = [utility / sum_utilities for utility in utilities]
        return np.var(x_utilities)

    def calculate_objectives(self, pop: np.ndarray, nodeID_RRs_set_dict: dict):
        objectives = np.zeros((pop.shape[0], self.M), dtype=float)
        for p_number in range(pop.shape[0]):
            p = list(pop[p_number])
            influcenced_RRs_set = self.get_influenced_RRs_set(p, nodeID_RRs_set_dict)
            ap, utilities, influence = self.influence(p, influcenced_RRs_set)
            objectives[p_number][0] = influence / self.graph.nodes_number
            objectives[p_number][1] = self.x_var(utilities)
        return objectives

    def make_new_fair_pop(self, F, B, nodeID_RRs_set_dict):
        Q_population = np.zeros((self.N, self.k), dtype=int)
        F0 = F[0]
        i = 0
        fc_ratio_dict = {0: self.fc_ratio, 1: 1 - self.fc_ratio}
        fm_p_dict = {0: self.fm_p, 1: 1 - self.fm_p}

        # 优先交叉
        while i < self.N:
            parents = []
            for x in range(2):
                rand = self.one_multinomial_exp(fc_ratio_dict)
                if rand == 0:
                    parents.append(random.choice(F0))
                else:
                    parents.append(random.choice(list(set([i for i in range(self.N)]).difference(set(parents)))))
            parents = self.crossover(parents)
            for p in parents:
                if i == self.N:
                    break
                else:
                    Q_population[i] = p
                    i += 1

        # 奖惩变异
        for p_number in range(len(Q_population)):
            rand = self.one_multinomial_exp(fm_p_dict)
            if rand == 0:
                Q_population[p_number] = self.fair_mutation(Q_population[p_number], B, nodeID_RRs_set_dict)
            else:
                Q_population[p_number] = self.mutation(Q_population[p_number])
        Q_population = np.unique(Q_population, axis=0)

        return Q_population

    def sort_by_distance(self, I, distance):
        dis = {}
        for i in I:
            dis[i] = distance[i]
        temp = sorted(dis.items(), key=lambda d: d[1])
        temp = [x[0] for x in temp]
        temp.reverse()
        return temp

    def f0fronts2resultdatalist(self, f0: list, objs: np.ndarray):
        r = []
        for index, p_number in enumerate(f0):
            r.append([index, list(self.population[p_number]), objs[p_number][0], objs[p_number][1]])
        return r

    def fair_nsga2(self, RRs_set, prior_seed_sets):
        start_time = time.time()
        self.init_population_prior_seed_set(prior_seed_sets)   # 种群初始化
        self.N_RRs_set = len(RRs_set) / self.graph.nodes_number  # 每个node有多少个RR_set
        nodeID_RRs_set_dict = self.get_nodeID_RRs_set_dict(RRs_set)

        B = self.gen_B(nodeID_RRs_set_dict)

        print('B created at', time.time() - start_time)
        for t in range(self.T):
            objs = self.calculate_objectives(self.population, nodeID_RRs_set_dict)
            # save prior_seed_sets's objs
            if t == 0:
                r_prior = self.f0fronts2resultdatalist(list(range(len(prior_seed_sets))), objs)
                Cal_HV.CalHv(r_prior, self.file, -1, self.T)

            F, rank = self.fast_non_dominated_sort(objs)

            r = self.f0fronts2resultdatalist(F[0], objs)
            Cal_HV.CalHv(r, self.file, t, self.T)

            distance = {}
            for f in F.keys():
                dis = self.crowding_distance(F[f], objs)
                distance = {**distance, **dis}

            Q_population = self.make_new_fair_pop(F, B, nodeID_RRs_set_dict)

            # 合并种群
            R_population = np.concatenate((self.population, Q_population))
            R_population = np.unique(R_population, axis=0)
            new_R_population = np.zeros([2 * self.N, self.k], dtype=int)
            for p_number in range(len(R_population)):
                new_R_population[p_number] = R_population[p_number]
            for p_number in range(len(R_population), 2 * self.N):
                p = random.sample(list([i for i in range(1, self.graph.nodes_number+1)]), self.k)
                p = np.array(p)
                new_R_population[p_number] = p
            R_population = new_R_population

            # 种群评估
            objectives = self.calculate_objectives(R_population, nodeID_RRs_set_dict)

            # 种群排序
            F, rank = self.fast_non_dominated_sort(objectives)   # 快速非支配排序
            i = 0
            j = 0
            distance = {}   #
            new_population = np.zeros([self.N, self.k], dtype=int)
            while j + len(F[i]) <= self.N:
                d = self.crowding_distance(F[i], objectives)
                distance.update(d)
                for p_number in F[i]:
                    new_population[j] = R_population[p_number]
                    j += 1
                i += 1

            d = self.crowding_distance(F[i], objectives)  # 拥挤度排序
            distance.update(d)
            F[i] = self.sort_by_distance(F[i], distance)
            for p_number in F[i][:self.N-j]:
                new_population[j] = R_population[p_number]
                j += 1
            self.population = new_population

            print('round', t, 'finished at', time.time() - start_time)

        with open('zjx_results/youtube_result/emo_seeds.txt', 'w') as f:
            for mm in range(len(F[0])):
                print(self.population[mm])
                f.write(str(self.population[mm]))
                f.write('\n')