from Measure import *
import math
import xlwt
# import xlrd
import os
from matplotlib import pyplot as plt
import math
class Save_Data:
    def __init__(self, path: str, k, prob):
        self.measures = []         # 元素为measure对象
        self.path = path           # 保存到的文件夹路径
        self.excel_name = 'k=' + str(k) + ',prob=' + str(prob) + '.xls'

    def delete_excel(self):
        path = self.path+self.excel_name
        if os.path.exists(path):  # 如果文件存在
            # 删除文件，可使用以下两种方法。
            os.remove(path)
            # os.unlink(path)
        else:
            print('no such file:'+path)  # 则返回文件不存在

    def save_in_excel(self):
        xls = xlwt.Workbook()
        sht = xls.add_sheet('sheet')
        sht.write(0, 0, 'algorithm name')
        sht.write(0, 1, 'normal seed set')
        sht.write(0, 2, 'fair seed set')
        sht.write(0, 3, 'normal influence')
        sht.write(0, 4, 'fair influence')
        sht.write(0, 5, 'POF')
        sht.write(0, 6, 'ROF')
        sht.write(0, 7, 'F-score')
        sht.write(0, 8, 'min cap')
        sht.write(0, 9, 'max cap')
        sht.write(0, 10, 'utility')
        for i in range(1,len(self.measures)+1):
            sht.write(i, 0, str(self.measures[i-1].algorithm_name))
            sht.write(i, 1, str(self.measures[i-1].normal_result.seed_set))
            sht.write(i, 2, str(self.measures[i-1].fair_result.seed_set))
            sht.write(i, 3, str(self.measures[i-1].normal_result.influence))
            sht.write(i, 4, str(self.measures[i-1].fair_result.influence))
            sht.write(i, 5, str(self.measures[i-1].POF))
            sht.write(i, 6, str(self.measures[i-1].ROF))
            sht.write(i, 7, str(self.measures[i-1].f_score))
            sht.write(i, 8, str(self.measures[i-1].min_cap))
            sht.write(i, 9, str(self.measures[i-1].max_cap))
            sht.write(i, 10, str(self.measures[i-1].utility))
        xls.save(self.path + self.excel_name)

