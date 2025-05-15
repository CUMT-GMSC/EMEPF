import json
import os.path


class CalHv:
    def __init__(self, f0fronts, path, t, T):
        self.f0fronts = f0fronts  # [index, population, objs[0], objs[1]]
        self.path = path
        self.t = t  # 当前轮
        self.T = T  # 总轮数
        self.population_list = list()
        self.hv = 0
        self.cal_hv()

    def cal_hv(self):
        ref_point = (0, 1)
        f0_points = [ref_point]
        for f0 in self.f0fronts:
            f0_points.append((f0[2], f0[3]))
            self.population_list.append(f0[1])
        f0_points = sorted(f0_points, key=lambda x: x[0])
        for i in range(1, len(f0_points)):
            self.hv += abs((f0_points[i][0]-f0_points[i-1][0])*(f0_points[i][1]-f0_points[0][1]))

        self.save_result()

    def save_result(self):
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                f.write('[')
        with open(self.path, 'a') as f:
            pop_objs_dict = {f0[0]: (f0[1], f0[2], f0[3]) for f0 in self.f0fronts}
            result = {'t': str(self.t + 1),
                      'pop_objs_dict': str(pop_objs_dict),
                      'hv': str(self.hv)}
            f.write(json.dumps(result, ensure_ascii=False, indent=4))
            if self.t + 1 < self.T:
                f.write(',')
            f.write('\r\n')
            if self.t + 1 == self.T:
                f.write(']')
