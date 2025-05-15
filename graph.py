
import networkx as nx
import matplotlib.pyplot as plt

# Node编号 1-N
# Community  1-M



class Node:
    def __init__(self, nodeID):
        self.nodeID = nodeID    # 正整数，所有节点的编号从1到N，N即节点数量
        self.status = 0   # status = 0 / 1， 0是未激活状态，1是激活状态


class Community:
    def __init__(self, cID , community):
        self.cID = cID      # 群体ID, 从1-M
        self.community = community     # list， list内元素为节点ID
        self.size = 0   # 初始化为0，群体内节点数量


class Communities:
    def __init__(self, communities,node_community_dic):
        self.communities = communities   # dic，  key-value      cID-community对象
        self.node_community_dic = node_community_dic  # dic， nodeID-cID 字典
        self.size = len(communities)



class graph:
    def __init__(self, graph, id_node_dic, Communities:Communities):
        self.graph = graph     # networkx图类对象
        self.Communities = Communities
        self.id_node_dic = id_node_dic
        self.nodes_number = graph.number_of_nodes()
        self.edges_number = graph.number_of_edges()
        self.nodes_id_list = self.nodeID_list()

    def ID_of_node(self,node):
        return node.nodeID

    def nodeID_list(self):
        return list(map(self.ID_of_node,self.graph.nodes()))

    def generate_id_node_dic(self):
        id_node_dic = {}
        for node in self.graph.nodes():
            id = node.nodeID
            id_node_dic[id] = node
        return id_node_dic

