import os
import csv

import pylab
from numpy import array, column_stack
from numpy.linalg import solve, matrix_rank, LinAlgError


class Stocks:
    def __init__(self, companies: dict[str, list[float]], profit: float = 0):
        self.__correct(data, profit)
        self.r = profit
        self.companies = companies
        self.interval = self.create_interval()

    def __correct(self, data_: dict[str, list[float]], r: float):
        if data_ is None:
            data_ = {}
        if not isinstance(r, float) and not isinstance(r, int):
            raise ValueError("r должно быть числом")
        if not isinstance(data_, dict):
            raise ValueError("Необходимо подавать словарь в формате {'str': []}")
        for i in list(data_.keys()):
            if not isinstance(i, str):
                raise ValueError("Ключи должны быть строками")
            if not isinstance(data_[i], list):
                raise ValueError("Элементы словаря должны быть списками ")
            for j in data_[i]:
                if not isinstance(j, float) and not isinstance(j, int):
                    raise ValueError("Элементы списка - целые и дробные числа")

    # График прибыли и долей распределения бюджета между компаниями
    def profit_graphic(self):
        values = self.analyze()
        profits = list(values.keys())
        shares = []
        for index in range(len(self.companies)):
            temp = []
            for share in list(values.values()):
                temp.append(share[index])
            shares.append(temp)

        companies = list(self.companies.keys())
        pylab.subplot(1, 3, 1)
        pylab.xlabel('Profit')
        pylab.ylabel('Shares')
        for i in range(len(shares)):
            pylab.plot(profits, shares[i], label="{}".format(companies[i]))
            pylab.legend()

    # Грифик прибыли и рисков
    def risk_graphic(self):
        values = self.analyze()
        profits = list(values.keys())
        risks = [self.risk_indicator(company) for company in self.companies]
        shares = []
        for index in range(len(self.companies)):
            temp = []
            for share in list(values.values()):
                temp.append(share[index])
            shares.append(temp)

        ratios = []
        for i in range(len(shares)):
            temp = []
            for share in shares[i]:
                temp.append(risks[i] * share)
            ratios.append(temp)

        companies = list(self.companies.keys())
        pylab.subplot(1, 3, 2)
        pylab.xlabel('Profit')
        pylab.ylabel('Risk')
        for i in range(len(ratios)):
            pylab.plot(profits, ratios[i], label="{}".format(companies[i]))
            pylab.legend()

    # График долей распредения между компаниями и риски
    def safe_graphic(self):
        values = self.analyze()
        risks = [self.risk_indicator(company) for company in self.companies]
        shares = []
        for index in range(len(self.companies)):
            temp = []
            for share in list(values.values()):
                temp.append(share[index])
            shares.append(temp)

        ratios = []
        for i in range(len(shares)):
            temp = []
            for share in shares[i]:
                temp.append(risks[i] * share)
            ratios.append(temp)

        companies = list(self.companies.keys())
        pylab.subplot(1, 3, 3)
        pylab.xlabel('Shares')
        pylab.ylabel('Risk')
        for i in range(len(ratios)):
            pylab.plot(shares[i], ratios[i], label="{}".format(companies[i]))
            pylab.legend()

    # Выодим все возможные значения параметра r.
    def analyze(self) -> dict[float, list[float]]:
        ws = {}
        for self.r in self.interval:
            try:
                w = self.count_shares().tolist()
            # Закоменчено условие, при котором все доли в отрезке от 0 до 1
            # flag = True
            # for i in w:
            #     if i < 0:
            #         flag = False
            #         break
            # if flag:
                ws[self.r] = w
            except AttributeError:
                continue

        return ws

    # Задаём интервал поиска значений для параметра r
    def create_interval(self) -> list[float]:
        rs = []
        for company in self.companies:
            rs.append(self.average(company))

        min_r = -1
        max_r = 1
        current_r, interval = min_r, []
        while current_r <= max_r:
            interval.append(current_r)
            current_r = round(current_r + 0.001, 3)

        return interval

    # Считаем риск конкретной компании
    def risk_indicator(self, company) -> float:
        values = self.companies[company]
        average = self.average(company)

        risk = 0
        for value in values:
            risk += (value - average) ** 2
        return round((risk / len(company)) ** 0.5, 3)

    # Выводим матрицу вместе с решениями (долями), округляем значения
    def count_shares(self):
        matrix = self.create_matrix()
        extend = [0 for _ in range(len(self.companies))]
        extend += [self.r, 1]

        old_matrix = array(matrix)
        old_extend = array(extend)
        new_matrix = []
        for line in old_matrix.tolist():
            new_matrix.append([round(el, 3) for el in line])
        new_extend = [str(round(i, 3)) for i in array(old_extend).tolist()]
        printable_extend = ['|' + str(i) for i in new_extend]

        for index in range(len(matrix)):
            new_matrix[index].append(printable_extend[index])
        print(*matrix, sep='\n')

        try:
            return solve(old_matrix, old_extend)[:-2]
        except LinAlgError:
            rang_matrix1 = matrix_rank(matrix)
            rang_matrix2 = matrix_rank(column_stack((matrix, extend)))
            if rang_matrix1 == rang_matrix2 and rang_matrix1 < matrix.shape[1]:
                return "Система имеет бесконечно много решений"
            else:
                return "Система не имеет решения"

    # Формируем матрицу
    def create_matrix(self):
        covs, rs = [], []
        for company1 in self.companies:
            temp = []
            for company2 in self.companies:
                temp.append(round(self.cov(company1, company2) * 2, 3))

            ri = self.average(company1)
            rs.append(ri)
            temp += [ri, 1]
            covs.append(temp)

        covs.append(rs + [0, 0])
        covs.append([1 for _ in range(len(self.companies))])
        covs[-1] += [0, 0]
        return array(covs)

    # Считаем коварицию (дисперсию, если компании одинаковы)
    def cov(self, company1: str, company2: str) -> float:
        numbers1 = self.companies[company1]
        numbers2 = self.companies[company2]
        x = self.average(company1)
        y = self.average(company2)

        summa, count = 0, 0
        for i in range(len(numbers1)):
            summa += (numbers1[i] - x) * (numbers2[i] - y)
            count += 1

        return round(summa / count, 3)

    # Считаем среднее арифметическое данных одной компании
    def average(self, company: str) -> float:
        numbers = self.companies[company]
        return round(sum(numbers) / len(numbers), 3)


# Делаем одинаковое количество дней для всех компаний
def formating(r: dict[str, list[float]]) -> dict[str, list[float]]:
    length = 10 ** 10
    for i in list(r.values()):
        new_length = len(i)
        length = min(new_length, length)

    return {company: values[:length] for company, values in r.items()}


def data_read(directory: str = 'stock'):  # Достаём данные из csv файликов
    files = os.listdir(directory)
    changes = {}
    for file in files:
        company = file[file.index('-') + 2:-4]
        with open(f'{directory}/{file}', 'r') as f:
            reader = csv.reader(f)
            count = 1
            rows = []
            for row in reader:
                if count:
                    count -= 1
                    continue
                rows.append(float(row[-1][:-1].replace(',', '.')))
            changes[company] = rows

    return formating(changes)


if __name__ == '__main__':
    # data = {
    #     'A': [-0.49, 0.41, 0.53, 0.98, 0.12, 0.19, 1.16, 1.65, -0.72, -1.75, 0.11, 1.58, -0.32, -1.52, -2.61, -0.31,
    #           0.62, -1.73, -2.03, -0.44, 1.24, -0.35, -0.19],
    #     'B': [-0.49, 0.41, 0.53, 0.98, 0.12, 0.19, 1.16, 1.65, -0.72, -1.75, 0.11, 1.58, -0.32, -1.52, -2.61, -0.31,
    #           0.62, -1.73, -2.03, -0.44, 1.24, -0.35, -0.19],
    #     'C': [-0.61, 0.70, -0.24, 1.50, -0.69, -0.28, 1.86, -1.05, -1.50, -1.03, 0.66, -1.34, -0.10, -0.31, -2.30,
    #           -0.68, 0.72, -2.31, -8.07, -5.30, 1.96, -2.82, 0.84]
    # }
    data = data_read()
    s = Stocks(data, 0.5)
    # print(s.count_shares())
    s.profit_graphic()
    s.risk_graphic()
    s.safe_graphic()
    pylab.show()
