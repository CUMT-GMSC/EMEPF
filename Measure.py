
from graph import *
import numpy as np
import time
import json
import os
from beautifulJson import NoIndent,MyEncoder


class resultData:
    def __init__(self,graphName:str,attribute:str, p:float, k:int, RRs_num:int, algorithmName:str,generation:int,f0fronts:list, cp = None, mp = None,fc_ratio = None,fm_p = None, compute_time = None):
        self.graphName = graphName
        self.attribute = attribute
        self.p = p
        self.k = k
        self.RRs_num = RRs_num
        self.algorithmName = algorithmName
        self.generation = generation
        # self.f0fronts = f0fronts      #[[种子集,utilities,influence,xvar],[种子集,utilities,influence,xvar]]
        self.f0fronts = f0fronts  # [[seed_set, influence, xvar], ...]
        self.cp = cp     #CFF中是concave function type
        self.mp = mp     #CFF中是function parameter
        self.fc_ratio = fc_ratio  # f0fronts 占 crossover的比例
        self.fm_p = fm_p  # 发生fair mutation的概率
        self.compute_time = compute_time
        self.saveTime = time.localtime()
        self.HV = self.HV()

    def HV(self):
        ref_point = (0,1)
        f0_points = [ref_point]
        for f0 in self.f0fronts:
            # f0_points.append((f0[2],f0[3]))
            f0_points.append((f0[1],f0[2]))
        f0_points = sorted(f0_points, key=lambda x: x[0])   # 升序
        hv = 0
        for i in range(1,len(f0_points)):
            hv+=abs((f0_points[i][0]-f0_points[i-1][0])*(f0_points[i][1]-f0_points[0][1]))
        return hv


    def NoIndentList(self,dic):
        for gen in dic:
            # l = dic[gen]['f0fronts']
            for i in range(len(dic[gen]['f0fronts'])):
                dic[gen]['f0fronts'][i] = NoIndent(dic[gen]['f0fronts'][i])
            dic[gen]['saveTime'] = NoIndent(dic[gen]['saveTime'])
        return dic



    def createFile(self,path):
        with open(path, 'w') as f:
            json.dump({}, f)

    def saveInJson(self,path):
        dic = self.saveInDic()
        l = {}
        if not os.path.exists(path):
            self.createFile(path)
        with open(path, "r", encoding="utf-8") as f:
            li = json.load(f)
            li['generation='+str(dic['generation'])] = dic
            l = li
        l = self.NoIndentList(l)
        with open(path, 'w') as f:
            json_data = json.dumps(l, cls=MyEncoder, ensure_ascii=False, sort_keys=True, indent=4)
            f.write(json_data)
            f.write('\n')

    @staticmethod
    def readJson(file, generation):
        with open(file) as f:
            j = json.load(f)
            return j[generation]['f0fronts']










