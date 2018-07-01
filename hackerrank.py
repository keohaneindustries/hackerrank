#!/bin/python3

# import math
# import os
# import random
# import re
# import sys
#
# from itertools import product


def maximize_it():
    def _maximize(m, el):

        from itertools import product
        return max([sum([j**2 for j in i])%m for i in product(*el)])
    
    km = input().split()
    
    k = int(km[0])
    
    m = int(km[1])
    
    el = []
    
    for _ in range(k):
        matrix_item = map(int, input().split())
        _=matrix_item.__next__()
        el.append(matrix_item)
    
    print(_maximize(m, el))


def basic_statistics1():
    # import numpy as np
    import pandas as pd
    _, ser = int(input()), pd.Series(list(map(int,input().split())))
    print(ser.mean())
    print(ser.median())
    print(ser.mode()[0])
    return


def basic_statistics2_weighted_mean_v1():
    import pandas as pd
    _, ser, wgts = int(input()), pd.Series(list(map(int, input().split()))), pd.Series(list(map(int, input().split())))
    print('%.1f' % (sum(ser*wgts)/sum(wgts)))
    return


def basic_statistics2_weighted_mean_v2():
    n, ser, wgts = int(input()), list(map(int, input().split())), list(map(int, input().split()))
    print('%.1f' % (sum([ser[i] * wgts[i] for i in range(n)]) / sum(wgts)))
    return


def arrays_DS():
    def _reverseArray(arr):
        import array
        
        arr = array.array('i', arr)
        arr.reverse()
        return arr.tolist()
    arr_count = int(input())
    
    arr = list(map(int, input().rstrip().split()))
    
    res = _reverseArray(arr)
    print(' '.join(map(str, res)))


def arrays_2D_DS():
    def _array2D(arr):
        result = 0
        rows = len(arr)-2
        cols = len(arr[0])-2
        for i in range(rows):
            for j in range(cols):
                cum=sum([sum(arr[i][j:j+3]),arr[i+1][j+1],sum(arr[i+2][j:j+3])])
                if i==0 and j==0:
                    result = cum
                elif cum>result:
                    result = cum
        return result

    arr = []
    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))

    result = _array2D(arr)
    print(result)
    return


def findthePoint():
    def _findPoint(px, py, qx, qy):
        x = qx-px
        y = qy-py
        return (x+qx,y+qy)
    
    n = int(input())

    for n_itr in range(n):
        pxPyQxQy = input().split()
    
        px = int(pxPyQxQy[0])
    
        py = int(pxPyQxQy[1])
    
        qx = int(pxPyQxQy[2])
    
        qy = int(pxPyQxQy[3])
    
        result = _findPoint(px, py, qx, qy)
    
        print(' '.join(map(str, result)))
    return


def numpy_arrays():
    import numpy
    def _arrays(arr):
        import array
        # arr = list(map(float,arr))
        # arr = array.array('f', arr)
        # arr.reverse()
        # result = numpy.array(arr, float)
        result = numpy.array(list(map(float, arr)), float)
        result=numpy.flip(result,0)
        return result
    
    arr = input().strip().split(' ')
    result = _arrays(arr)
    print(result)


def postal_codes():
    regex_integer_in_range = r'\b\d\d\d\d\d\d\b'  # Do not delete 'r'.
    regex_alternating_repetitive_digit_pair = r'(?P<num>\d)(?=\d\1)'  # Do not delete 'r'.

    import re
    
    P = input()
    
    print(bool(re.match(regex_integer_in_range, P))
          and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)
    return


def complex_numbers():
    import math

    class Complex(object):
    
        def __init__(self, real, imaginary):
            self.real = real
            self.imaginary = imaginary
    
        def __add__(self, no):
            return Complex(self.real + no.real, self.imaginary + no.imaginary)
    
        def __sub__(self, no):
            return Complex(self.real - no.real, self.imaginary - no.imaginary)
    
        def __mul__(self, no):
            return Complex((self.real * no.real - self.imaginary * no.imaginary),
                           (self.imaginary * no.real + self.real * no.imaginary))
    
        def __truediv__(self, no):
            return Complex(((self.real * no.real + self.imaginary * no.imaginary) / (no.real ** 2 + no.imaginary ** 2)),
                           ((self.imaginary * no.real - self.real * no.imaginary) / (no.real ** 2 + no.imaginary ** 2)))
    
        def mod(self):
            return Complex(math.sqrt(self.real**2+self.imaginary**2),0)
            
    
        def __str__(self):
            if self.imaginary == 0:
                result = "%.2f+0.00i" % (self.real)
            elif self.real == 0:
                if self.imaginary >= 0:
                    result = "0.00+%.2fi" % (self.imaginary)
                else:
                    result = "0.00-%.2fi" % (abs(self.imaginary))
            elif self.imaginary > 0:
                result = "%.2f+%.2fi" % (self.real, self.imaginary)
            else:
                result = "%.2f-%.2fi" % (self.real, abs(self.imaginary))
            return result
    
    c = map(float, input().split())
    d = map(float, input().split())
    x = Complex(*c)
    y = Complex(*d)
    print(*map(str, [x + y, x - y, x * y, x / y, x.mod(), y.mod()]), sep='\n')
    return


def torsional_angle():
    import math
    class Points(object):
        
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
        
        def __sub__(self, no):
            return Points((self.x-no.x), (self.y- no.y), (self.z-no.z))
        
        def dot(self, no):
            return ((self.x*no.x)+(self.y*no.y)+(self.z*no.z))
        
        def cross(self, no):
            return Points((self.y*no.z-self.z*no.y),(self.z*no.x-self.x*no.z),(self.x*no.y-self.y*no.x))
        
        def absolute(self):
            return pow((self.x ** 2 + self.y ** 2 + self.z ** 2), 0.5)

    points = list()
    for i in range(4):
        a = list(map(float, input().split()))
        points.append(a)

    a, b, c, d = Points(*points[0]), Points(*points[1]), Points(*points[2]), Points(*points[3])
    x = (b - a).cross(c - b)
    y = (c - b).cross(d - c)
    angle = math.acos(x.dot(y) / (x.absolute() * y.absolute()))

    print("%.2f" % math.degrees(angle))
    
    return


def read_list():
    n = int(input())
    l=[]
    
    for _ in range(n):
        item = input().split()
        if item[0]=="print":
            print(l)
        elif item[0] == "sort":
            l.sort()
        elif item[0] == "pop":
            l.pop()
        elif item[0] == "reverse":
            l.reverse()
        elif item[0] == "insert":
            l.insert(int(item[1]),int(item[2]))
        elif item[0] == "append":
            l.append(int(item[1]))
        elif item[0] == "remove":
            l.remove(int(item[1]))


def game_of_stones():
    import math
    import os
    import random
    import re
    import sys
    
    def gameOfStones(n):
        def move(n):
            P={}
            for i in [2,3,5]:
                if (n-i)>0:
                    P[i] = move((n-i))
                elif (n - i) == 0:
                    P[i] = "win"
                else:
                    P[i] = "lose"
            loss_cnt=0
            for i in [2,3,5]:
                if P[i] == "lose":
                    loss_cnt+=1
            if loss_cnt == 3:
                return "win"
            else:
                return "lose"
        if move(n) == "win":
            return "Second"
        else:
            return "First"
    
    t = int(input())
    for t_itr in range(t):
        n = int(input())
        result = gameOfStones(n)
        print(result)
    
    return


def towers():
    return


def run_binomial():
    import math
    
    def _binomial(p, n, hits):
        odds = 0
        for x in hits:
            f = math.factorial(n) / (math.factorial(x) * math.factorial((n - x)))
            odds += f * ((p) ** (x)) * ((1 - p) ** (n - x))
        return odds

    arr = list(map(int, input().split()))
    p = float(arr[0]) / 100
    n = arr[1]

    hits_atmost = list(range(0, 2 + 1))
    hits_atleast = list(range(0, 1 + 1))
    odds_atmost = _binomial(p, n, hits_atmost)
    odds_atleast = (1 - _binomial(p, n, hits_atleast))
    print('%.3f' % odds_atmost)
    print('%.3f' % odds_atleast)
    
    return


def run_negative_binomial():
    import math
    
    def _negative_binomial(x, p, hits):
        odds = 0
        for n in hits:
            f = math.factorial(n - 1) / (math.factorial(x - 1) * math.factorial(((n - 1) - (x - 1))))
            odds += f * ((p) ** (x)) * ((1 - p) ** (n - x))
        return odds

    arr = list(map(int, input().split()))
    p = float(arr[0]) / float(arr[1])
    x = 1
    n = int(input())
    hits_exactly = list(range(n, n + 1))

    odds_exactly = _negative_binomial(x, p, hits_exactly)

    print('%.3f' % odds_exactly)
    
    return


def get_even_odd_strings():
    
    def _even_odd_strings(s):
        s1 = []
        s2 = []
        for i, c in enumerate(s):
            if i == 0:
                s1.append(c)
            elif i % 2 == 1:
                s2.append(c)
            else:
                s1.append(c)
        snew = " ".join(["".join(s1), "".join(s2)])
        return snew
    
    n = int(input())
    for _ in range(n):
        s = input().rstrip()
        print(_even_odd_strings(s))
    
    return
    

def n_choose_x(n,x):
    import math
    return math.factorial(n) / (math.factorial(x) * math.factorial((n - x)))


def run_poisson():
    import math
    def _poisson(lam,ks):
        odds = 0
        for k in ks:
            odds += lam**k * math.e ** (-lam) / math.factorial(k)
        return odds

    lam = float(input())
    k = int(input())
    hits_exactly = list(range(k, k + 1))

    odds_exactly = _poisson(lam, hits_exactly)

    print('%.3f' % odds_exactly)

    return


def run_poisson2():
    import math
    def _poisson(lam,ks):
        odds = 0
        for k in ks:
            odds += lam**k * math.e ** (-lam) / math.factorial(k)
        return odds
    
    def _poisson_squared(lam):
        return lam + lam**2
    
    def _exp_cost(a0, a1,lam):
        return a0+a1*_poisson_squared(lam)
    
    lams = list(map(float,input().split()))
    lama = lams[0]
    lamb = lams[1]
    print(_exp_cost(160,40,lama))
    print(_exp_cost(128,40,lamb))
    
    return


def run_normal():
    import math
    
    def _normal_t(u,sd,t):
        a = (1/2)*(1+(sd*math.sqrt(2*math.pi)))
        b = math.e ** (-((t-u)**2/(2*sd**2)))
        odds = a * b
        return odds
    
    nd = list(map(float,input().split()))
    u = nd[0]
    sd = nd[1]
    t = float(input())
    ci = list(map(float,input().split()))
    ci_min = ci[0]
    ci_max = ci[1]
    
    print('%.3f' % _normal_t(u,sd,t))
    print('%.3f' % (_normal_t(u,sd,ci_min)-_normal_t(u,sd,ci_max)))
    
    return


def run_normal2():
    import math
    def _run_normal(u,sd,t):
        return (1/2)*(1+math.erf((t-u) /(sd*math.sqrt(2))))

    params = list(map(float,input().split()))

    u = params[0]
    sd = params[1]
    t_high = float(input())
    t_fail = float(input())

    print('%.2f' % ((1-_run_normal(u,sd,t_high))*100))
    print('%.2f' % ((1-_run_normal(u,sd,t_fail))*100))
    print('%.2f' % ((_run_normal(u,sd,t_fail))*100))

    return


def clt1():
    import math
    def _run_normal(u,sd,t):
        return (1/2)*(1+math.erf((t-u) /(sd*math.sqrt(2))))

    t_fail = float(input())
    n = float(input())
    u = float(input())
    sd = float(input())

    print('%.4f' % _run_normal(u*n,sd*math.sqrt(n),t_fail))

    return


def clt2():
    import math
    def _run_normal(u,sd,t):
        return (1/2)*(1+math.erf((t-u) /(sd*math.sqrt(2))))

    t_fail = float(input())
    n = float(input())
    u = float(input())
    sd = float(input())

    print('%.4f' % _run_normal(u*n,sd*math.sqrt(n),t_fail))

    return


def clt3():
    import math
    
    n = float(input())
    mean = float(input())
    std = float(input())
    ci = float(input())
    zScore = float(input())
    marginOfError = zScore * std / math.sqrt(n);
    print(mean - marginOfError)
    print(mean + marginOfError)
    return


def corr1():
    import math
    
    def _get_sd(l, u):
        sd = math.sqrt(sum([(l[i] - u) ** 2 for i in range(len(l))]) / (len(l)))
        return sd
    
    def _get_corr(n, x_l, y_l):
        x_u = sum(x_l) / n
        y_u = sum(y_l) / n
        
        x_sd = _get_sd(x_l, x_u)
        y_sd = _get_sd(y_l, y_u)
        
        num = sum([(x_l[i] - x_u) * (y_l[i] - y_u) for i in range(n)])
        denom = (n * x_sd * y_sd)
        return num / denom
    
    n = int(input())
    x_l = list(map(float, input().split()))
    y_l = list(map(float, input().split()))
    
    print('%.3f' % _get_corr(n, x_l, y_l))
    
    return


def spearman_corr():
    def _get_rank(X, n):
        x_rank = dict((x, i + 1) for i, x in enumerate(sorted(set(X))))
        return [x_rank[x] for x in X]
    
    n = int(input())
    X = list(map(float, input().split()))
    Y = list(map(float, input().split()))
    
    rx = _get_rank(X, n)
    ry = _get_rank(Y, n)
    
    d = [(rx[i] - ry[i]) ** 2 for i in range(n)]
    rxy = 1 - (6 * sum(d)) / (n * (n * n - 1))
    
    print('%.3f' % rxy)


def MLR():
    # import data
    import numpy as np
    
    m, n = [int(i) for i in input().strip().split(' ')]
    X = []
    Y = []
    for i in range(n):
        data = input().strip().split(' ')
        X.append(data[:m])
        Y.append(data[m:])
    q = int(input().strip())
    X_new = []
    for x in range(q):
        X_new.append(input().strip().split(' '))
    X = np.array(X, float)
    Y = np.array(Y, float)
    X_new = np.array(X_new, float)

    # center
    X_R = X - np.mean(X, axis=0)
    Y_R = Y - np.mean(Y)

    # calculate beta
    beta = np.dot(np.linalg.inv(np.dot(X_R.T, X_R)), np.dot(X_R.T, Y_R))

    # predict
    X_new_R = X_new - np.mean(X, axis=0)
    Y_new_R = np.dot(X_new_R, beta)
    Y_new = Y_new_R + np.mean(Y)

    # print
    for i in Y_new:
        print(round(float(i), 2))
    return


def LSR():
    x = [95, 85, 80, 70, 60]
    y = [85, 95, 70, 65, 70]
    ax = sum(x) / len(x)
    ay = sum(y) / len(y)
    sx = 0
    n = 0
    d = 0
    for i in range(len(x)):
        n += (ax - x[i]) * (ay - y[i])
        d += (ax - x[i]) ** 2
    slope = (1.0 * n) / d
    inter = ay - slope * ax
    y = slope * 80 + inter
    print(round(y, 3))
    return


def binary():
    n = int(input())
    over = True
    db = 1
    digts = 1
    while over:
        if n <=db:
            over = False
        else:
            db = db * 2
            digts +=1
    op = 0
    rem = n
    for i in range(digts):
        cur = digts-i-1
        dig = rem//(2**cur)
        op += dig*(10**cur)
        rem = rem % (2**cur)
    
    op=str(op)

    streak = 0
    max_streak = 0
    for i in op:
        if i =='1':
            streak +=1
            if streak>max_streak:
                max_streak = streak
        else:
            streak = 0
    
    print(str(max_streak))
    return


def hourglass():
    arr = []
    
    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))
    
    max_sum = -999
    for x in range(4):
        for y in range(4):
            cur_sum = arr[y][x]+arr[y][x+1]+arr[y][x+2]+arr[y+1][x+1]+arr[y+2][x]+arr[y+2][x+1]+arr[y+2][x+2]
            if cur_sum > max_sum:
                max_sum = cur_sum
            
    print(str(max_sum))
    
    return

def inheritance():
    ### START of fixed code
    class Person:
        
        def __init__(self, firstName, lastName, idNumber):
            self.firstName = firstName
            self.lastName = lastName
            self.idNumber = idNumber
        
        def printPerson(self):
            print("Name:", self.lastName + ",", self.firstName)
            print("ID:", self.idNumber)

    class Student(Person):
    ### END of fixed code
    
        def __init__(self,firstName, lastName, idNumber,scores):
            super().__init__(firstName, lastName, idNumber)
            self.scores = scores

        def calculate(self):
            avg = sum(self.scores) / len(self.scores)
            if (avg >= 90) and (avg <= 100):
                grade = 'O'
            elif (avg >= 80) and (avg <= 90):
                grade = 'E'
            elif (avg >= 70) and (avg <= 80):
                grade = 'A'
            elif (avg >= 55) and (avg <= 70):
                grade = 'P'
            elif (avg >= 40) and (avg <= 55):
                grade = 'D'
            else:
                grade = 'T'
            return grade
    
    
    ### START of fixed code
    line = input().split()
    firstName = line[0]
    lastName = line[1]
    idNum = line[2]
    numScores = int(input())  # not needed for Python
    scores = list(map(int, input().split()))
    s = Student(firstName, lastName, idNum, scores)
    s.printPerson()
    print("Grade:", s.calculate())
    
    return


def abstract_classes():
    ### START of fixed code
    from abc import ABCMeta, abstractmethod
    
    class Book(object, metaclass=ABCMeta):
        
        def __init__(self, title, author):
            self.title = title
            self.author = author
        
        @abstractmethod
        def display(): pass

    ### END of fixed code

    class MyBook(Book):
        def __init__(self,title, author, price):
            super().__init__(title, author)
            self.price = price
            
        def display(self):
            print("Title: "+ self.title)
            print("Author: "+ self.author)
            print("Price: "+ str(self.price))


    ### START of fixed code
    title = input()
    author = input()
    price = int(input())
    new_novel = MyBook(title, author, price)
    new_novel.display()
    
    return



def main():
    abstract_classes()
    return


if __name__ == '__main__':
    main()