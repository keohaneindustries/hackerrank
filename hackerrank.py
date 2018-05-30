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


def main():
    run_poisson()
    return


if __name__ == '__main__':
    main()