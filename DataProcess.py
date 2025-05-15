import math
import xlwt
import os
from matplotlib import pyplot as plt
import math
class result:
    def __init__(self,function_parameter,standard_seed,fair_seed,standard_variance,fair_variance,standard_influence_spread,fair_influence_spread):
        self.function = function_parameter
        self.standard_seed = standard_seed    # 标准种子集
        self.fair_seed = fair_seed       # 公平种子集
        self.standard_variance = standard_variance    # 标准下的节点ap的方差
        self.fair_variance = fair_variance        # 公平下的节点ap的方差
        self.standard_influence_spread = standard_influence_spread       # 标准下的影响力扩展度
        self.fair_influence_spread = fair_influence_spread        # 公平下的影响力扩展度
        self.ROF = 1-math.sqrt(self.fair_variance)/math.sqrt(self.standard_variance)    # Reward of Fairness  使用标准差，小数开根号更大一点，看着更明显
        self.POF = 1-self.fair_influence_spread/self.standard_influence_spread        # Price of Fairness
        self.RPR = self.ROF/self.POF          # ROF POF Ratio


class FunctionResults:
    def __init__(self,function_type,path):
        self.function_type = function_type    # 字符串，记录凸函数的类型，如幂函数，指数函数，对数函数
        self.results = []              # list，元素为result对象
        self.path = path         #  文件存放路径文件夹


    def DeleteExcel(self):
        path = self.path+'result.xls'
        if os.path.exists(path):  # 如果文件存在
            # 删除文件，可使用以下两种方法。
            os.remove(path)
            # os.unlink(path)
        else:
            print('no such file:'+path )  # 则返回文件不存在

    def ReadExcel(self):
        workbook = xlrd.open_workbook(self.path+'result.xls')
        sheet = workbook.sheets()[0]
        nrows = sheet.nrows
        ncols = sheet.ncols
        for i in range(1, nrows):
            function = float(sheet.cell_value(i, 0))
            standard_seed = eval(sheet.cell_value(i, 1))
            fair_seed = eval(sheet.cell_value(i, 2))
            standard_variance = float(sheet.cell_value(i, 3))
            fair_variance = float(sheet.cell_value(i, 4))
            standard_influence_spread = float(sheet.cell_value(i, 5))
            fair_influence_spread = float(sheet.cell_value(i, 6))
            ROF = float(sheet.cell_value(i, 7))
            POF = float(sheet.cell_value(i, 8))
            RPR = float(sheet.cell_value(i, 9))
            r = result(function, standard_seed, fair_seed, standard_variance, fair_variance, standard_influence_spread,
                       fair_influence_spread)
            self.results.append(r)


    def SaveInExcel(self):
        xls = xlwt.Workbook()
        sht = xls.add_sheet('sheet')
        sht.write(0, 0, 'function_param')
        sht.write(0, 1, 'standard_seed')
        sht.write(0, 2, 'fair_seed')
        sht.write(0, 3, 'standard_variance')
        sht.write(0, 4, 'fair_variance')
        sht.write(0, 5, 'standard_influence_spread')
        sht.write(0, 6, 'fair_influence_spread')
        sht.write(0, 7, 'ROF')
        sht.write(0, 8, 'POF')
        sht.write(0, 9, 'RPR')
        for i in range(len(self.results)):
            sht.write(i+1, 0, str(self.results[i].function))
            sht.write(i+1, 1, str(self.results[i].standard_seed))
            sht.write(i+1, 2, str(self.results[i].fair_seed))
            sht.write(i+1, 3, str(self.results[i].standard_variance))
            sht.write(i+1, 4, str(self.results[i].fair_variance))
            sht.write(i+1, 5, str(self.results[i].standard_influence_spread))
            sht.write(i+1, 6, str(self.results[i].fair_influence_spread))
            sht.write(i+1, 7, str(self.results[i].ROF))
            sht.write(i+1, 8, str(self.results[i].POF))
            sht.write(i+1, 9, str(self.results[i].RPR))
        xls.save(self.path+'result.xls')



    # data_type 1:ROF     2:POF    3:RPR
    def vs_data(self,data_type):
        N = len(self.results)
        X = []
        Y = []
        for i in range(N):
            X.append(self.results[i].function)
            if data_type == 1:
                Y.append(self.results[i].ROF)
            elif data_type == 2:
                Y.append(self.results[i].POF)
            else:
                Y.append(self.results[i].RPR)
        Xmin = min(X)
        Xmax = max(X)
        Ymin = min(Y)
        Ymax = max(Y)
        fig = plt.figure()
        plt.xlim((Xmin,Xmax))
        plt.ylim((Ymin,Ymax))
        plt.title(str(self.function_type))
        plt.xlabel('function param')
        plt.plot(X, Y, color='red', linewidth=2.0, linestyle='--')
        if data_type == 1:
            plt.ylabel('ROF')
            plt.savefig(self.path+'ROF.png')
        elif data_type == 2:
            plt.ylabel('POF')
            plt.savefig(self.path+'POF.png')
        else:
            plt.ylabel('RPR')
            plt.savefig(self.path+'RPR.png')
        plt.show()


