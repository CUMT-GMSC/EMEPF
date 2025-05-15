import copy
import math
import random
from multiprocessing import Queue,Manager,Process
import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
from graph import *
from LoadData import edgeList2Graph
from Measure import *
import heapq

class IMM(object):
    def __init__(self,graph,k,approx_rate,err_pr,graph_type):
        self.graph = graph     #Graph类
        self.k = k
        self.approx_rate = approx_rate
        self.err_pr = err_pr
        self.probability = 0
        self.graph_type = graph_type  # 有向图为1，无向图为0



    def Edge_live_or_not(self, probability):
        return int(np.random.binomial(1, probability, 1))


    def neighbors(self, Graph, nodeID, neighbor_type):

        neighbors = []
        node = Graph.id_node_dic[nodeID]
        if self.graph_type == 0:
            neighbors = list(Graph.graph.neighbors(node))
        elif neighbor_type == 0:
            neighbors = list(Graph.graph.predecessors(node))
        elif neighbor_type == 1:
            neighbors = list(Graph.graph.successors(node))
        elif neighbor_type == 2:
            neighbors = list(Graph.graph.predecessors(node))+list(Graph.graph.successors(node))
        return neighbors


    def refresh_graph(self):
        for node in self.graph.graph.nodes():
            node.status = 0




    # def generate_Live_Edge_Graph(self):
    #     print('-----generate Live_Edge_Graph-----')
    #
    #     Live_Edge_graph = nx.DiGraph()
    #     nodes = {}
    #
    #     for edge in self.graph.graph.edges():
    #         probability = self.graph.graph[edge[0]][edge[1]]['probability']
    #         if self.Edge_live_or_not(probability)==1:
    #             for i in range(2):
    #                 if edge[i].nodeID not in nodes:
    #                     nodes[edge[i].nodeID] = Node(edge[i].nodeID)
    #             Live_Edge_graph.add_edge(nodes[edge[0].nodeID],nodes[edge[1].nodeID])
    #     return Graph(Live_Edge_graph,nodes)

    def node_to_ID(self,node):
        return node.nodeID

    # def generate_Reverse_Reachable_set(self , Live_Edge_Graph , nodeID):
    #     print('-----granerate Reverse_Reachable_set----- ')
    #     Reverse_Reachable_set = [nodeID]    # 存放的是ID
    #     id_node_dic = Live_Edge_Graph.id_node_dic
    #     for ID in Reverse_Reachable_set:
    #         node = id_node_dic[ID]
    #         predecessors = self.neighbors(Live_Edge_Graph, node, 0)   # node节点的所有predecessors
    #         predecessors = list(map(self.node_to_ID,predecessors))
    #         for id in predecessors:
    #             if id in Reverse_Reachable_set:
    #                 predecessors.remove(id)
    #         Reverse_Reachable_set += predecessors
    #     return Reverse_Reachable_set


    def influence_to_neighbors(self,  nodeID,influence_type):
        graph = self.graph
        influenced_neighbors = []
        node = graph.id_node_dic[nodeID]
        for neighbor in self.neighbors(graph, nodeID,influence_type):
            if neighbor.status == 1:
                continue
            probability = self.probability
            influenced = self.Edge_live_or_not(probability)  # 节点是否能够影响邻居，能1 ，不能0
            if influenced == 1:
                influenced_neighbors.append(neighbor.nodeID)
                neighbor.status = 1  # 邻居的状态刷新，要么维持0，要么变成1
        return influenced_neighbors


    def generate_RRs(self,nodeID):
        graph = self.graph
        seed = self.graph.id_node_dic[nodeID]
        seed.status = 1
        RRs = [nodeID]
        for nodeID in RRs:
            influenced_neighbors = self.influence_to_neighbors(nodeID,0)
            RRs += influenced_neighbors
        self.refresh_graph()
        return RRs

    def diffuse(self,seed_set):

        influenced_nodesID = copy.deepcopy(seed_set)
        for nodeID in seed_set:
            node = self.graph.id_node_dic[nodeID]
            node.status = 1
        for nodeID in influenced_nodesID:
            influenced_neighbors = self.influence_to_neighbors(nodeID,1)
            influenced_nodesID += influenced_neighbors
        self.refresh_graph()
        return influenced_nodesID

    def repeat_nodes(self,nodes):
        n = set(nodes)
        if len(n) < len(nodes):
            return True
        else:
            return False


    def cover_num(self,S,RRs_set):
        cover_num = 0
        covered = [False for n in range(len(RRs_set))]
        for nodeID in S:
            for i in range(len(RRs_set)):
                if nodeID in RRs_set[i][2:] and covered[i]==False:
                    cover_num +=1
                    covered[i] =True
        return cover_num

    def concave_function(self , function_type,function_parameter,x):
        if function_type == 0:
            return math.log2(math.pow(x, function_parameter)+1)
        elif function_type == 1:
            return math.pow(x, function_parameter)
        elif function_type == 2:
            return 1-math.exp(-function_parameter*x)
        elif function_type == 3:
            return math.pow(function_parameter*x-x*x, 0.5)
        elif function_type == 4:
            return math.sin(math.pow(x, function_parameter)*math.pi/2)
        elif function_type == 5:
            return 1-function_parameter*(x-1)*(x-1)
        elif function_type == 6:
            return math.pow(x, 0.1) - function_parameter * x
        else:
            return 0

    def function_name(self,function_type,function_parameter):
        if function_type == 0:
            return 'log2(x^'+str(function_parameter)+'+1)'
        elif function_type == 1:
            return 'x^'+str(function_parameter)
        elif function_type == 2:
            return '1-e^(-'+str(function_parameter)+'*x)'
        else:
            return 0


    def set_edge_prob(self, probability):
        self.probability = probability
        if self.graph_type==0:
            if probability == -1:
                for u,v in self.graph.graph.edges():
                    self.graph.graph[u][v]['probability'] = 2/(self.graph.graph.degree(u)+self.graph.graph.degree(v))
            else:
                for u,v in self.graph.graph.edges():
                    self.graph.graph[u][v]['probability'] = probability
        else:
            if probability == -1:
                for node in self.graph.graph.nodes():
                    for predecessors in self.graph.graph.predecessors(node):
                        self.graph.graph[predecessors][node]['probability'] = 1/self.graph.graph.in_degree(node)

            else:
                for node in self.graph.graph.nodes():
                    for predecessors in self.graph.graph.predecessors(node):
                        self.graph.graph[predecessors][node]['probability'] = probability

    def node_outdegree(self):
        degree = [0] * (self.graph.nodes_number+1)
        id_node_dic = self.graph.id_node_dic
        for nodeID in range(1,self.graph.nodes_number+1):
            degree[nodeID] = self.graph.graph.out_degree(id_node_dic[nodeID])
        return degree

    def RRs2ap(self,seed_set:list,RRs_set):

        c = np.zeros((self.graph.nodes_number + 1, self.graph.nodes_number + 1))
        N = len(RRs_set) / self.graph.nodes_number
        node_RRs_dic = {}
        covered = {}
        covered_RRs_num = {}
        ap = [0] * (self.graph.nodes_number + 1)
        for i in range(1, self.graph.nodes_number + 1):
            node_RRs_dic[i] = []
        for index, RRs in enumerate(RRs_set):
            covered[RRs[0]] = False
            covered_RRs_num[RRs[1]] = 0
            for nodeID in RRs[1:]:
                c[nodeID][RRs[1]] += 1
                node_RRs_dic[nodeID].append(RRs)

        for seed in seed_set:
            for nodeID in range(1, self.graph.nodes_number + 1):
                ap[nodeID] += (c[seed][nodeID]) / N
            for nodeID in range(1, self.graph.nodes_number + 1):
                covered_RRs_num[nodeID] += c[seed][nodeID]
            for R in node_RRs_dic[seed]:
                if not covered[R[0]]:
                    covered[R[0]] = True
                    for u in R[1:]:
                        c[u][R[1]] -= 1
        return ap

    def ap2utility(self, ap):
        utility = [0] * (self.graph.Communities.size + 1)
        communities = self.graph.Communities
        for cID in communities.communities.keys():
            for nodeID in communities.communities[cID].community:
                utility[cID] += ap[nodeID]
            utility[cID] /= communities.communities[cID].size
        return utility

    def RRs2ap2(self, seed_set: list, RRs_set, N):
        # 已知种子集的情况下，不用更新RRs_set
        # N = len(RRs_set) / self.graph.nodes_number
        ap = [float(0)] * (self.graph.nodes_number + 1)
        for RRs in RRs_set:
            if set(RRs[1:]).intersection(set(seed_set)):
                ap[RRs[1]] += 1
        for i in range(1, self.graph.nodes_number + 1):
            ap[i] /= N
        return ap

    def RRs_set2groupRRs_set(self,RRs_set):
        A = []
        B = []
        V = set([i for i in range(1,self.graph.nodes_number+1)])
        for i in range(self.graph.nodes_number+1):
            A.append([])
        for i in range(self.graph.Communities.size+1):
            B.append([])
        for RRs in RRs_set:
            if len(RRs)>2:
                A[RRs[1]] = list(set(A[RRs[1]]) | set(RRs[2:]))
        for v in range(1,self.graph.nodes_number+1):
            c = self.graph.Communities.node_community_dic[v]
            B[c] = list(set(B[c]) | set(A[v]))
        for cID in range(1,self.graph.Communities.size+1):
            V = V - set(B[cID])
        B[0] = list(V)
        return B

    def B(self,RRs_set):
        A = []
        B = []
        V = set([i for i in range(1, self.graph.nodes_number + 1)])
        for i in range(self.graph.nodes_number + 1):
            A.append([])
        for i in range(self.graph.Communities.size + 1):
            B.append([])
        for RRs in RRs_set:
            if len(RRs) > 2:
                A[RRs[1]] = list(set(A[RRs[1]]) | set(RRs[2:]))
        for v in range(1, self.graph.nodes_number + 1):
            c = self.graph.Communities.node_community_dic[v]
            B[c] = list(set(B[c]) | set(A[v]))
        for cID in range(self.graph.Communities.size + 1):
            for x in range(cID+1,self.graph.Communities.size + 1):
                B[cID] = list(set(B[cID]).difference(set(B[x])))
        for cID in range(1, self.graph.Communities.size + 1):
            V = V - set(B[cID])
        B[0] = list(V)
        return B


    def estimate_RRs_num(self):
        RRs_set = []
        LB = 1
        approx_rate_ = math.pow(self.approx_rate,0.5)
        o = [0]
        #print('开始估计要生成的反向可达集数量！！')
        #print('估计影响力扩展度')
        for i in range(1, math.ceil(math.log(self.graph.nodes_number, 2))):
            # print('-----影响力扩展度迭代中-----')
            x = self.graph.nodes_number / (math.pow(2, i))
            o.append(math.ceil((self.graph.nodes_number * (2 + approx_rate_ * 2 / 3) * (
                        math.log(math.comb(self.graph.nodes_number, self.k)) + self.err_pr * math.log(
                    self.graph.nodes_number) + math.log(2) + math.log(math.log(self.graph.nodes_number, 2)))) / (
                                           approx_rate_ * approx_rate_ * x)))
            print('----------------需要生成', math.ceil(o[i] - o[i - 1]), '个反向可达集-------------------------')
            for j in range(math.ceil(o[i] - o[i - 1])):
                random_node_ID = random.randint(1, self.graph.nodes_number )
                RRs = self.generate_RRs(random_node_ID)
                RRs_set.append(RRs)
            print('-----当前反向可达集数为', len(RRs_set), '-----')
            r = copy.deepcopy(RRs_set)
            for index, RRs in enumerate(r):
                RRs.insert(0, index)
            S = self.NodeSelection(r)
            influence_spread = self.graph.nodes_number * self.cover_num(S, r) / (len(r))
            if influence_spread >= (1 + approx_rate_) * x:
                LB = influence_spread / (1 + approx_rate_)
                # print('-----当前影响力扩展度为', LB, '-----')
                break
        #print('-----当前反向可达集数为', len(RRs_set), '-----')
        # print('--------------------估计影响力扩展度为：', LB)

        a = math.sqrt(self.err_pr * math.log(self.graph.nodes_number) + math.log(4))
        b = math.sqrt((1 - 1 / math.e) * (math.log(math.comb(self.graph.nodes_number, self.k)) + self.err_pr * math.log(
            self.graph.nodes_number) + math.log(4)))
        length_of_RRs_set = math.ceil((2 * self.graph.nodes_number * math.pow((1 - 1 / math.e) * a + b, 2)) / (
                    LB * self.approx_rate * self.approx_rate))
        #print('------------------要生成   ', length_of_RRs_set, '   个反向可达集')
        return length_of_RRs_set

    # 每个节点生成N个反向可达集
    def fair_Sampling(self,N):
        RRs_set = []
        for nodeID in range(1,1+self.graph.nodes_number):
            for i in range(N):
                RRs = self.generate_RRs(nodeID)
                RRs_set.append(RRs)
                # print('-----当前反向可达集数为', len(RRs_set), '-----')
        return RRs_set

    def single_fair_Sampling(self,q:Queue,N):
        RRs_set = []
        for nodeID in range(1,1+self.graph.nodes_number):
            for i in range(N):
                RRs = self.generate_RRs(nodeID)
                RRs_set.append(RRs)

        q.put(RRs_set,block=False)


    # 大师兄要实现的统计一个节点会由多少个节点到达的采样函数
    def temp_Sampling(self,N):
        t = 0
        RRs_set = []
        RRs_set.append([])
        for nodeID in range(1,1+self.graph.nodes_number):
            R = []
            for i in range(N):
                RRs = self.generate_RRs(nodeID)
                R += RRs
                t+=1
                # print('-----当前反向可达集数为', len(RRs_set), '-----')
            RRs_set.append(R)
        print(t)
        return RRs_set

    def fair_NodeSelection(self,RRs_set,concave_function_type,function_parameter):
        # 二维数组 , c[v][r]指的是v节点覆盖r节点的反向可达集数量
        c = np.zeros((self.graph.nodes_number+1,self.graph.nodes_number+1))
        N = len(RRs_set)/self.graph.nodes_number
        node_RRs_dic = {}  # 节点ID：节点所覆盖的反向可达集序列
        covered = {}
        covered_RRs_num = {}    # covered_RRs_num[nodeID]            key为nodeID，value为根为nodeID的被覆盖反向可达集数量
        seed_set = []
        temp_fair_influence = [0]*(self.graph.nodes_number+5)
        # 初始化上面的这些数据
        for index,RRs in enumerate(RRs_set):
            covered[RRs[0]] = False
            covered_RRs_num[RRs[1]] = 0
            for nodeID in RRs[2:]:
                if nodeID not in node_RRs_dic:
                    node_RRs_dic[nodeID] = []
                c[nodeID][RRs[1]] += 1
                node_RRs_dic[nodeID].append(RRs)
        # 贪心算法，k轮寻找k个种子
        for i in range(self.k):
            # 计算各个节点作为种子时的公平影响力扩展度，并存到temp_fair_influence中，共n个元素，下标是节点id，value是该节点作为种子时的公平影响力扩展度
            for nodeID in range(1,self.graph.nodes_number+1):
                if nodeID not in seed_set:
                    if concave_function_type == -1:
                        ap = [0] * (self.graph.nodes_number+1)
                        for r in range(1, self.graph.nodes_number+1):
                            ap[r] = (c[nodeID][r] + covered_RRs_num[r]) / N
                        temp_fair_influence[nodeID] = sum(ap)
                        temp_fair_influence[nodeID] = (1 - function_parameter) * temp_fair_influence[nodeID] - function_parameter * self.graph.nodes_number * np.var(ap)
                    else:
                        for r in range(1,self.graph.nodes_number+1):
                            temp_fair_influence[nodeID] += self.concave_function(concave_function_type,function_parameter ,(c[nodeID][r]+covered_RRs_num[r])/N)




            new_seed = temp_fair_influence.index(max(temp_fair_influence))  #得到新种子
            if new_seed == 0:
                continue
            seed_set.append(new_seed)   #将新种子加入到种子集中
            #print('新fair种子是：',new_seed)
            temp_fair_influence = [0]*(self.graph.nodes_number+5)  #将这个list清零，供下一轮循环使用
            #更新covered_RRs_num
            for nodeID in range(1,self.graph.nodes_number+1):
                covered_RRs_num[nodeID] += c[new_seed][nodeID]
            for R in node_RRs_dic[new_seed]:
                if not covered[R[0]]:
                    covered[R[0]] = True
                    for u in R[2:]:
                        if u!=new_seed:
                            c[u][R[1]] -= 1
        return seed_set

    def CELF_fair_NodeSelection(self,RRs_set,concave_function_type,function_parameter):
        # 二维数组 , c[v][r]指的是v节点覆盖r节点的反向可达集数量
        c = np.zeros((self.graph.nodes_number+1,self.graph.nodes_number+1))
        N = len(RRs_set)/self.graph.nodes_number
        node_RRs_dic = {}  # 节点ID：节点所覆盖的反向可达集序列
        covered = {}
        covered_RRs_num = {}    # covered_RRs_num[nodeID]            key为nodeID，value为根为nodeID的被覆盖反向可达集数量
        seed_set = []
        temp_fair_influence = [0]*(self.graph.nodes_number+5)
        # 初始化上面的这些数据
        for index,RRs in enumerate(RRs_set):
            covered[RRs[0]] = False
            covered_RRs_num[RRs[1]] = 0
            for nodeID in RRs[2:]:
                if nodeID not in node_RRs_dic:
                    node_RRs_dic[nodeID] = []
                c[nodeID][RRs[1]] += 1
                node_RRs_dic[nodeID].append(RRs)
        # 贪心算法，k轮寻找k个种子
        qs = 0   #记录σ(s)
        for i in range(self.k):
            # 计算各个节点作为种子时的公平影响力扩展度，并存到temp_fair_influence中，共n个元素，下标是节点id，value是该节点作为种子时的公平影响力扩展度
            new_seed = 0
            if i==0:
                for nodeID in range(1, self.graph.nodes_number + 1):
                    if nodeID not in seed_set:
                        for r in range(1, self.graph.nodes_number + 1):
                            temp_fair_influence[nodeID] += self.concave_function(concave_function_type,function_parameter, (c[nodeID][r] +covered_RRs_num[r]) / N)
                new_seed = temp_fair_influence.index(max(temp_fair_influence))  # 得到新种子
                #更新σ(s)
                qs = temp_fair_influence[new_seed]
            else:
                flag = [0]*(self.graph.nodes_number+5)
                qs1 = [0]*(self.graph.nodes_number+5)
                #最多会循环n次，所以设置for循环n次，其实中间他选到种子后就会提前break，远远不会循环到n次
                for v in range(self.graph.nodes_number):
                    candidate_seed = temp_fair_influence.index(max(temp_fair_influence))
                    if flag[candidate_seed] == 1:
                        new_seed = candidate_seed
                        break
                    else:
                        temp_fair_influence[candidate_seed] = 0
                        for r in range(1,self.graph.nodes_number+1):
                            temp_fair_influence[candidate_seed] += self.concave_function(concave_function_type,function_parameter, (c[candidate_seed][r] +covered_RRs_num[r]) / N)
                        qs1[candidate_seed] = temp_fair_influence[candidate_seed]
                        temp_fair_influence[candidate_seed] -= qs
                        flag[candidate_seed] = 1
                qs = qs1[new_seed]

            #上面已经选完种子了，下面开始更新各种list
            # 将种子节点的temp_fair_influence设置为-1，以避免在后面会重复选到这个种子
            temp_fair_influence[new_seed] = -1
            seed_set.append(new_seed)  # 将新种子加入到种子集中
            # 更新covered_RRs_num
            for nodeID in range(1, self.graph.nodes_number + 1):
                covered_RRs_num[nodeID] += c[new_seed][nodeID]
            for R in node_RRs_dic[new_seed]:
                if not covered[R[0]]:
                    covered[R[0]] = True
                    for u in R[2:]:
                        if u != new_seed:
                            c[u][R[1]] -= 1
        return seed_set

    def CELF_community_fair_NodeSelection(self,RRs_set,concave_function_type,function_parameter):

        c = np.zeros((self.graph.nodes_number+1,self.graph.nodes_number+1),dtype='float32')
        N = len(RRs_set)/self.graph.nodes_number
        node_RRs_dic = {}
        covered = {}
        covered_RRs_num = {}
        seed_set = []
        temp_fair_influence = [0]*(self.graph.nodes_number+5)

        for i in range(1,self.graph.nodes_number+1):
            node_RRs_dic[i] = []
        for index,RRs in enumerate(RRs_set):
            covered[RRs[0]] = False
            covered_RRs_num[RRs[1]] = 0
            for nodeID in RRs[1:]:
                c[nodeID][RRs[1]] += 1
                node_RRs_dic[nodeID].append(RRs)
        qs = 0
        for i in range(self.k):
            new_seed = 0
            if i == 0:
                for nodeID in range(1, self.graph.nodes_number + 1):
                    for cID in self.graph.Communities.communities.keys():
                        community_ap_sum = 0
                        community = self.graph.Communities.communities[cID]
                        for r in community.community:
                            community_ap_sum += (c[nodeID][r] + covered_RRs_num[r]) / N
                        cp = community_ap_sum / community.size
                        temp_fair_influence[nodeID] += community.size * self.concave_function(concave_function_type, function_parameter , cp)
                new_seed = temp_fair_influence.index(max(temp_fair_influence))
                qs = temp_fair_influence[new_seed]
            else:
                flag = [0]*(self.graph.nodes_number+5)
                qs1 = [0]*(self.graph.nodes_number+5)
                for v in range(2 * self.graph.nodes_number):
                    candidate_seed = temp_fair_influence.index(max(temp_fair_influence))
                    if flag[candidate_seed] == 1:
                        new_seed = candidate_seed
                        break
                    else:
                        temp_fair_influence[candidate_seed] = 0
                        for cID in self.graph.Communities.communities.keys():
                            community_ap_sum = 0
                            community = self.graph.Communities.communities[cID]
                            for r in community.community:
                                community_ap_sum += (c[candidate_seed][r] + covered_RRs_num[r]) / N
                            cp = community_ap_sum / community.size
                            temp_fair_influence[candidate_seed] += community.size * self.concave_function(concave_function_type,function_parameter,cp)
                        qs1[candidate_seed] = temp_fair_influence[candidate_seed]
                        temp_fair_influence[candidate_seed] -= qs
                        flag[candidate_seed] = 1
                qs = qs1[new_seed]
            temp_fair_influence[new_seed] = -1
            seed_set.append(new_seed)
            for nodeID in range(1, self.graph.nodes_number + 1):
                covered_RRs_num[nodeID] += c[new_seed][nodeID]
            for R in node_RRs_dic[new_seed]:
                if not covered[R[0]]:
                    covered[R[0]] = True
                    for u in R[1:]:
                        c[u][R[1]] -= 1
        for i in range(len(seed_set)):
            seed_set[i] = int(seed_set[i])
        return seed_set


    def Sampling(self ,length_of_RRs_set ):
        RRs_set = []
        for i in range(length_of_RRs_set):
            random_node_ID = random.randint(1, self.graph.nodes_number)
            RRs = self.generate_RRs(random_node_ID)
            RRs_set.append(RRs)
        for index,RRs in enumerate(RRs_set):
            RRs.insert(0, index)
        return RRs_set


    def NodeSelection(self,RRs_set):
        c = np.zeros(self.graph.nodes_number + 1)
        node_RRs_dic = {}
        covered = {}
        seed_set = []
        for index,RRs in enumerate(RRs_set):
            covered[RRs[0]] = False
            for nodeID in RRs[1:]:
                if nodeID not in node_RRs_dic:
                    node_RRs_dic[nodeID] = []
                c[nodeID] += 1
                node_RRs_dic[nodeID].append(RRs)
        for i in range(self.k):
            v = np.argmax(c)
            seed_set.append(v)
            for R in node_RRs_dic[v]:
                if not covered[R[0]]:
                    covered[R[0]] = True
                    for u in R[1:]:
                        # if u!=v:
                        #     if u in c:
                        c[u] -= 1
        for i in range(len(seed_set)):  # seed set里面是int64类型，json不能存该类型，所以强制转化为int型
            seed_set[i] = int(seed_set[i])
        if self.repeat_nodes(seed_set):
            print('出现重复节点！！！！！')
        return seed_set

    def max_avg_degree(self):
        degree = np.zeros(self.graph.nodes_number + 1, dtype=int)
        for node in self.graph.graph.nodes():
            nodeID = node.nodeID
            degree[nodeID] = self.graph.graph.degree(node)
        max_degree = max(degree)
        avg_degree = np.mean(degree)
        return max_degree,avg_degree








    def degree_concave_nodeselection(self,RRs_set,concave_function_type,function_parameter,a):
        '''

        :param RRs_set:
        :param concave_function_type:
        :param function_parameter:
        :param a: 候选种子占全体节点的百分比
        :return:
        '''
        degree = self.node_outdegree()
        # 二维数组 , c[v][r]指的是v节点覆盖r节点的反向可达集数量
        c = np.zeros((self.graph.nodes_number + 1, self.graph.nodes_number + 1), dtype=int)
        N = len(RRs_set) / self.graph.nodes_number
        node_RRs_dic = {}  # 节点ID：节点所覆盖的反向可达集序列
        covered = {}
        covered_RRs_num = {}  # covered_RRs_num[nodeID]            key为nodeID，value为根为nodeID的被覆盖反向可达集数量
        seed_set = []
        # 初始化上面的这些数据
        for i in range(1, self.graph.nodes_number + 1):
            node_RRs_dic[i] = []
        for index, RRs in enumerate(RRs_set):
            covered[RRs[0]] = False
            covered_RRs_num[RRs[1]] = 0
            for nodeID in RRs[2:]:
                c[nodeID][RRs[1]] += 1
                node_RRs_dic[nodeID].append(RRs)
        # 贪心算法，k轮寻找k个种子
        qs = 0  # 记录fair σ(s)
        for i in range(self.k):
            temp_fair_influence = [0] * (self.graph.nodes_number + 5)
            # 计算各个节点作为种子时的公平影响力扩展度，并存到temp_fair_influence中，共n个元素，下标是节点id，value是该节点作为种子时的公平影响力扩展度
            candidate_seeds = list(map(degree.index, heapq.nlargest((math.ceil((self.graph.nodes_number-i+1)*a)), degree)))
            for nodeID in candidate_seeds:
                if nodeID not in seed_set:
                    for cID in self.graph.Communities.communities.keys():
                        community_ap_sum = 0  # 社区的ap值的和
                        community = self.graph.Communities.communities[cID]  # 当前社区对象
                        for r in community.community:
                            community_ap_sum += (c[nodeID][r] + covered_RRs_num[r]) / N
                        cp = community_ap_sum / community.size  # 社区平均激活概率
                        temp_fair_influence[nodeID] += community.size * self.concave_function(concave_function_type,
                                                                                              function_parameter,
                                                                                              cp)  # PR效用函数
            new_seed = temp_fair_influence.index(max(temp_fair_influence))  # 得到新种子
            degree[new_seed] = 0
            # 更新fair σ(s)
            qs = temp_fair_influence[new_seed]
            # 上面已经选完种子了，下面开始更新各种list
            # 将种子节点的temp_fair_influence设置为-1，以避免在后面会重复选到这个种子
            seed_set.append(new_seed)  # 将新种子加入到种子集中
            # 更新covered_RRs_num
            for nodeID in range(1, self.graph.nodes_number + 1):
                covered_RRs_num[nodeID] += c[new_seed][nodeID]
            for R in node_RRs_dic[new_seed]:
                if not covered[R[0]]:
                    covered[R[0]] = True
                    for u in R[2:]:
                        if u != new_seed:
                            c[u][R[1]] -= 1
        return seed_set, qs  # 返回种子集和utility



    # 蒙特卡洛模拟
    def MC_InfEst(self,seed_set,times):
        ap = [0]*(self.graph.nodes_number+1)
        total_influence = 0
        for i in range(times):
            influence = self.diffuse(seed_set)
            total_influence += len(influence)
            for nodeID in influence:
                ap[nodeID] += 1

        return [*map(lambda x:x/times, ap)], total_influence/times

    def single_process_MC_InfEst(self,Q:Queue,seed_set,times):
        ap = [0] * (self.graph.nodes_number + 1)
        total_influence = 0
        for i in range(times):
            influence = self.diffuse(seed_set)
            total_influence += len(influence)
            for nodeID in influence:
                ap[nodeID] += 1
        Q.put([[*map(lambda x: x / times, ap)], total_influence / times], block=False)

    def multi_process_MC_InfEst(self,edge_list,cpu_core,seed_set,times):
        Q = Manager().Queue()
        process_list = []
        ap = np.array([0] * (self.graph.nodes_number + 1))
        influence = 0
        for i in range(cpu_core):
            G, Nodes = edgeList2Graph(edge_list)
            GRAPH = graph(G, Nodes, self.graph.Communities)
            imm = IMM(GRAPH, self.k, self.approx_rate, self.err_pr,self.graph_type)
            imm.set_edge_prob(self.probability)
            p = Process(target=imm.single_process_MC_InfEst, args=(Q,seed_set,math.ceil(times/cpu_core)))
            process_list.append(p)
        for p in process_list:
            p.start()
        for p in process_list:
            p.join()
        while Q.qsize() != 0:
            r = Q.get_nowait()

            ap = np.sum([ap,r[0]],axis=0)
            influence += r[1]
        ap = ap/cpu_core
        influence = influence/cpu_core
        return ap,influence

    @staticmethod
    def seed_set_similarity(seed_set1: list, seed_set2: list):
        x = set(seed_set1) & set(seed_set2)
        x = list(x)
        return len(x)/len(seed_set1)

    @staticmethod
    def seedset_in_seedsets(seed_set, seed_sets):
        '''
        :param seed_set: 种子集
        :param seed_sets: 种子集的集
        :return: 如果存在为True，反之  False
        '''
        for s in seed_sets:
            if IMM.seed_set_similarity(seed_set,s) == 1:
                return True
        return False


