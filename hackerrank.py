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


def scope():
    ### START of fixed code
    class Difference:
        
        def __init__(self, a):
            self.__elements = a

    ### END of fixed code
        
        def computeDifference(self):
            max_diff = 0
            elements = self.__elements
            for i in range(len(elements)-1):
                for j in range(i+1,len(elements)):
                    abs_diff = abs(elements[i]-elements[j])
                    if abs_diff > max_diff:
                        max_diff = abs_diff
            self.maximumDifference = max_diff


    ### START of fixed code
    _ = input()
    a = [int(e) for e in input().split(' ')]

    d = Difference(a)
    d.computeDifference()

    print(d.maximumDifference)
    
    return


def linked_lists():
    ### START of fixed code
    class Node:
    
        def __init__(self, data):
            self.data = data
            self.next = None

    class Solution:
    
        def display(self, head):
            current = head
            while current:
                print(current.data, end=' ')
                current = current.next
        
    ### END of fixed code
        def insert(self, head, data):
            if head is None:
                idk = Node(data)
                return idk
            else:
                current = head
                while current:
                    prev = current
                    current = current.next
                idk = Node(data)
                prev.next = idk
                return head


    ### START of fixed code
    mylist = Solution()
    T = int(input())
    head = None
    for i in range(T):
        data = int(input())
        head = mylist.insert(head, data)
    mylist.display(head)
    return


def exceptions_1():
    import sys
    
    S = input().strip()
    
    try:
        i = int(S)
        print(i)
    except(ValueError):
        print("Bad String")
    
    return


def exceptions_2():
    
    class Calculator(object):
        
        def power(self,n,p):
            if (n<0) or (p<0):
                raise Exception("n and p should be non-negative")
            else:
                return n**p
            
    ### START of fixed code
    myCalculator = Calculator()
    T = int(input())
    for i in range(T):
        n, p = map(int, input().split())
        try:
            ans = myCalculator.power(n, p)
            print(ans)
        except Exception as e:
            print(e)
    return
    

def queues():
    ### START of fixed code
    import sys

    class Solution:
    ### END of fixed code
        
        def __init__(self):
            self.stack = []
            self.queue = []
        
        def enqueueCharacter(self,char):
            self.queue.insert(0,char)
        
        def pushCharacter(self,char):
            self.stack.append(char)
        
        def popCharacter(self):
            return self.stack.pop()
        
        def dequeueCharacter(self):
            return self.queue.pop()
    
    
    ### START of fixed code
    # read the string s
    s = input()
    # Create the Solution class object
    obj = Solution()

    l = len(s)
    # push/enqueue all the characters of string s to stack
    for i in range(l):
        obj.pushCharacter(s[i])
        obj.enqueueCharacter(s[i])

    isPalindrome = True
    '''
    pop the top character from stack
    dequeue the first character from queue
    compare both the characters
    '''
    for i in range(l // 2):
        if obj.popCharacter() != obj.dequeueCharacter():
            isPalindrome = False
            break
    # finally print whether string s is palindrome or not.
    if isPalindrome:
        print("The word, " + s + ", is a palindrome.")
    else:
        print("The word, " + s + ", is not a palindrome.")
    
    return


def interfaces():
    ### START of fixed code
    class AdvancedArithmetic(object):
    
        def divisorSum(n):
            raise NotImplementedError
    ### END of fixed code
    
    import math
    class Calculator(AdvancedArithmetic):

        def __init__(self):
            self.divisors = []

        def divisorSum(self, n):
            self.max_divisor = int(math.floor(math.sqrt(n)))
            for x in range(1,self.max_divisor+1):
                if (n%x)==0:
                    self.divisors.append(x)
                    if x!=(n/x):
                        self.divisors.append(n/x)
            return int(sum(self.divisors))
        
        
    ### START of fixed code
    n = int(input())
    my_calculator = Calculator()
    s = my_calculator.divisorSum(n)
    print("I implemented: " + type(my_calculator).__bases__[0].__name__)
    print(s)


def bubble_sort():
    ### START of fixed code
    import sys
    
    n = int(input().strip())
    a = list(map(int, input().strip().split(' ')))
    ### END of fixed code
    
    def swap(a,ix):
        item=a.pop(ix)
        a.insert(ix+1,item)
        return a
        
    numSwaps = 0
    for i in range(n):
        for j in range(n-1):
            if (a[j] > a[j + 1]):
                numSwaps+=1
                a = swap(a,j)



    firstElement = a[0]
    lastElement = a[len(a)-1]
    print("Array is sorted in "+str(numSwaps) +" swaps.")
    print("First Element: "+str(firstElement))
    print("Last Element: "+str(lastElement))
    
    return


def binary_search_tree():
    ### START of fixed code
    class Node:
    
        def __init__(self, data):
            self.right = self.left = None
            self.data = data

    class Solution:
    
        def insert(self, root, data):
            if root == None:
                return Node(data)
            else:
                if data <= root.data:
                    cur = self.insert(root.left, data)
                    root.left = cur
                else:
                    cur = self.insert(root.right, data)
                    root.right = cur
            return root
    ### END of fixed code
        def getHeight(self, root):
            if root == None:
                return -1
            else:
                check_l = self.getHeight(root.left)
                check_r = self.getHeight(root.right)
                
                return 1 + max(check_l,check_r)
                
    ### START of fixed code
    T = int(input())
    myTree = Solution()
    root = None
    for i in range(T):
        data = int(input())
        root = myTree.insert(root, data)
    height = myTree.getHeight(root)
    print(height)
    
    return


def binary_search_tree2():
    ### START of fixed code
    import sys

    class Node:
    
        def __init__(self, data):
            self.right = self.left = None
            self.data = data

    class Solution:
    
        def insert(self, root, data):
            if root == None:
                return Node(data)
            else:
                if data <= root.data:
                    cur = self.insert(root.left, data)
                    root.left = cur
                else:
                    cur = self.insert(root.right, data)
                    root.right = cur
            return root
    ### END of fixed code

        def levelOrder(self, root):
            queue = [root] if root else []

            while queue:
                node = queue.pop()
                print(node.data, end=" ")

                if node.left: queue.insert(0, node.left)
                if node.right: queue.insert(0, node.right)
        
    
    ### START of fixed code
    T = int(input())
    myTree = Solution()
    root = None
    for i in range(T):
        data = int(input())
        root = myTree.insert(root, data)
    myTree.levelOrder(root)
    
    return


def primality():
    import math
    
    class AdvancedArithmetic(object):
        
        def divisorSum(n):
            raise NotImplementedError
    
    class Calculator(AdvancedArithmetic):
        
        def __init__(self):
            self.divisors = []
        
        def is_prime(self, n):
            self.max_divisor = int(math.floor(math.sqrt(n)))
            if n == 1:
                return "Not prime"
            elif n ==2:
                return "Prime"
            for x in range(2, self.max_divisor + 1):
                if (n % x) == 0:
                    return "Not prime"
                    
            return "Prime"
        
    T = int(input())
    my_calculator = Calculator()
    for i in range(T):
        n = int(input())
        print(my_calculator.is_prime(n))


def linked_lists_2():
    ### START of fixed code
    class Node:
        
        def __init__(self, data):
            self.data = data
            self.next = None
    
    class Solution:
        
        def insert(self, head, data):
            p = Node(data)
            if head == None:
                head = p
            elif head.next == None:
                head.next = p
            else:
                start = head
                while (start.next != None):
                    start = start.next
                start.next = p
            return head
        
        def display(self, head):
            current = head
            while current:
                print(current.data, end=' ')
                current = current.next

    ### END of fixed code
        def removeDuplicates(self,head):
            if not head:
                return head
            elif not head.next:
                return head
            else:
                if head.next.data == head.data:
                    head.next = head.next.next
                    self.removeDuplicates(head)
                else:
                    self.removeDuplicates(head.next)
                return head
    
    ### START of fixed code
    mylist = Solution()
    T = int(input())
    head = None
    for i in range(T):
        data = int(input())
        head = mylist.insert(head, data)
    head = mylist.removeDuplicates(head)
    mylist.display(head)
    return


def nested_logic():
    da, ma, ya = map(int, input().split())
    de, me, ye = map(int, input().split())
    
    if ya > ye:
        print(10000)
    elif ya < ye:
        print(0)
    else:
        if ma < me:
            print(0)
        elif ma == me:
            if da <= de:
                print(0)
            else:
                print(15*(da-de))
        else:
            print(500 * (ma - me))
        
    
    return


def testing():
    ### START of fixed code
    import array
    def minimum_index(seq):
        if len(seq) == 0:
            raise ValueError("Cannot get the minimum value index from an empty sequence")
        min_idx = 0
        for i in range(1, len(seq)):
            if seq[i] < seq[min_idx]:
                min_idx = i
        return min_idx
    
    ### END of fixed code

    class TestDataEmptyArray(object):
        
        @staticmethod
        def get_array():
            import array
            return array.array('i')

    class TestDataUniqueValues(object):
    
        @staticmethod
        def get_array():
            import array
            l= [-1,5,8,7,0]
            return array.array('i',l)
    
        @staticmethod
        def get_expected_result():
            return 0

    class TestDataExactlyTwoDifferentMinimums(object):
    
        @staticmethod
        def get_array():
            import array
            l = [-1, 5, 8, 7, 0,-1]
            return array.array('i', l)
    
        @staticmethod
        def get_expected_result():
            return 0
    
    ### START of fixed code
    def TestWithEmptyArray():
        try:
            seq = TestDataEmptyArray.get_array()
            result = minimum_index(seq)
        except ValueError as e:
            pass
        else:
            assert False

    def TestWithUniqueValues():
        seq = TestDataUniqueValues.get_array()
        assert len(seq) >= 2
    
        assert len(list(set(seq))) == len(seq)
    
        expected_result = TestDataUniqueValues.get_expected_result()
        result = minimum_index(seq)
        assert result == expected_result

    def TestiWithExactyTwoDifferentMinimums():
        seq = TestDataExactlyTwoDifferentMinimums.get_array()
        assert len(seq) >= 2
        tmp = sorted(seq)
        assert tmp[0] == tmp[1] and (len(tmp) == 2 or tmp[1] < tmp[2])
    
        expected_result = TestDataExactlyTwoDifferentMinimums.get_expected_result()
        result = minimum_index(seq)
        assert result == expected_result

    TestWithEmptyArray()
    TestWithUniqueValues()
    TestiWithExactyTwoDifferentMinimums()
    print("OK")


def regex():
    import math
    import os
    import random
    import re
    import sys
    
    if __name__ == '__main__':
        N = int(input())
        names = []
        for N_itr in range(N):
            firstNameEmailID = input().split()
            
            firstName = firstNameEmailID[0]
            
            emailID = firstNameEmailID[1]
            
            if re.search('(?<=@)gmail\.com$',emailID):
                names.append(firstName)
        
        names.sort()
        for name in names:
            print(name)


def bitwise():
    import math
    import os
    import random
    import re
    import sys
    
    if __name__ == '__main__':
        T = int(input().strip())
        for _ in range(T):
            n, k = map(int, input().split())
            print(k - 1 if ((k - 1) | k) <= n else k - 2)
    return


def time_conversion():
    import os
    import sys
    
    def timeConversion(s):
        t = s[:-2]
        tm = s[-2:]
        hr,mn,sc = map(int,t.split(':'))
        if tm == 'PM':
            if hr!=12:
                hr+=12
        elif hr ==12:
            hr-=12
        return '{}:{}:{}'.format('{:02d}'.format(hr),'{:02d}'.format(mn),'{:02d}'.format(sc))
        
    
    s = input()
    result = timeConversion(s)
    print(result)


def leaderboard():
    import math
    import os
    import random
    import re
    import sys
    
    # Complete the climbingLeaderboard function below.
    def climbingLeaderboard(scores, alice):
        scores.sort()
        result = []
        p=len(scores)
        i0 = 0
        r0 = p+1
        
        for s in alice:
            i1 = max([0,i0-1])
            r1 = p-i1+1
            while i1 < p:
                if s < scores[i1]:
                    i0 = i1
                    i1 = p
                elif s == scores[i1]:
                    r1 -= 1
                    i1 += 1
                    i0 = i1
                    i1 = p
                elif s > scores[i1]:
                    r1-=1
                    i1 += 1
            result.append(r1)
            r0=r1
        return result
    
    def remove_dupes(l):
        i=0
        while i < (len(l)-1):
           if l[i] == l[i+1]:
               w=l.pop(i)
           else:
               i+=1
        return l
    
    scores_count = int(input())
    
    scores = list(map(int, input().rstrip().split()))
    scores = remove_dupes(scores)
    
    alice_count = int(input())
    
    alice = list(map(int, input().rstrip().split()))
    
    result = climbingLeaderboard(scores, alice)
    
    for res in result:
        print(res)
    

def magic_square():
    import math
    import os
    import random
    import re
    import sys

    def formingMagicSquare(s):
        
        def check_rows(s):
            if (sum(s[0])==sum(s[1])) and (sum(s[1])==sum(s[2])):
                return True
            else:
                return False
        def check_cols(s):
            if (sum([s[0][0],s[1][0],s[2][0]])==sum([s[0][1],s[1][1],s[2][1]])) and (sum([s[0][2],s[1][2],s[2][2]])==sum([s[0][1],s[1][1],s[2][1]])):
                return True
            else:
                return False
        def check_diags(s):
            if sum([s[0][0],s[2][2]])==sum([s[0][2],s[2][0]]):
                return True
            else:
                return False
        def check_all(s):
            if check_rows(s):
                if check_cols(s):
                    if check_diags(s):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        
        
        
        return s
    
    s = []
    for _ in range(3):
        s.append(list(map(int, input().rstrip().split())))

    result = formingMagicSquare(s)
    
    print(str(result))
    
    return


def kangaroos():
    def kangaroo(x1, v1, x2, v2):
        b = x1-x2
        a = v1-v2
        if a==0:
            if b==0:
                return "YES"
            else:
                return "NO"
        else:
            x = (0-b)/a
            if (x>=0) and ((x%1)==0):
                return "YES"
            else:
                return "NO"
    
    x1V1X2V2 = input().split()
    
    x1 = int(x1V1X2V2[0])
    
    v1 = int(x1V1X2V2[1])
    
    x2 = int(x1V1X2V2[2])
    
    v2 = int(x1V1X2V2[3])
    
    result = kangaroo(x1, v1, x2, v2)
    print(result)


def min_abs_diff_in_array():
    def minimumAbsoluteDifference(arr):
        arr.sort()
        arr2=arr[1:]
        l = []
        for i in range(len(arr2)):
            l.append(arr2[i]-arr[i])
        
        return min(l)

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = minimumAbsoluteDifference(arr)
    print(str(result))
    
    return


def luck_balance():
    def luckBalance(k, contests):
        luck = 0
        important = []
        for c in contests:
            if c[1]==1:
                important.append(c[0])
            else:
                luck+=c[0]
        if k==0:
            luck -= sum(important)
            return luck
        else:
            important.sort()
            luck+=sum(important[-k:])
            luck-=sum(important[:-k])
            return luck

    nk = input().split()

    n = int(nk[0])

    k = int(nk[1])

    contests = []

    for _ in range(n):
        contests.append(list(map(int, input().rstrip().split())))

    result = luckBalance(k, contests)
    print(str(result))


def mark_and_toys():
    import math
    import os
    import random
    import re
    import sys
    def maximumToys(prices, k):
        prices.sort()
        result = 0
        for p in prices:
            if k >=p:
                result+=1
                k-=p
            else:
                break
        return result
    
    nk = input().split()
    
    n = int(nk[0])
    
    k = int(nk[1])
    
    prices = list(map(int, input().rstrip().split()))
    
    result = maximumToys(prices, k)
    print(str(result))
    return


def priyanka_and_toys():
    import math
    import os
    import random
    import re
    import sys
    
    def toys(w):
        w.sort()
        current_min = 0
        result =0
        for wgt in w:
            if (current_min==0) and (result==0):
                result += 1
                current_min=wgt
            elif wgt > (current_min+4):
                result += 1
                current_min = wgt
        return result
    
    n = int(input())
    
    w = list(map(int, input().rstrip().split()))
    
    result = toys(w)
    print(str(result))
    return


def cutting_boards():
    import math
    import os
    import random
    import re
    import sys
    
    # Complete the boardCutting function below.
    def boardCutting(cost_y, cost_x):
        total_cost = 0
        min_cost = 0
        return (min_cost % (10**9+7))

    q = int(input())

    for q_itr in range(q):
        mn = input().split()
    
        m = int(mn[0])
    
        n = int(mn[1])
    
        cost_y = list(map(int, input().rstrip().split()))
    
        cost_x = list(map(int, input().rstrip().split()))
    
        result = boardCutting(cost_y, cost_x)
        print(str(result))
    
    
    return


def permuting_two_arrays():
    def twoArrays(k, A, B):
        A.sort()
        B.sort(reverse=True)
        if (any(a + b < k for a, b in zip(A, B))):
            return "NO"
        else:
            return "YES"

    q = int(input())

    for q_itr in range(q):
        nk = input().split()
    
        n = int(nk[0])
    
        k = int(nk[1])
    
        A = list(map(int, input().rstrip().split()))
    
        B = list(map(int, input().rstrip().split()))
    
        result = twoArrays(k, A, B)
        
        print(result)
    
    return


def jim_and_the_orders():
    print(*(lambda s: sorted(range(1, len(s) + 1), key=lambda i: s[i - 1]))(
        tuple(sum(map(int, input().split())) for _ in range(int(input())))))
    return


def max_min():
    # Complete the maxMin function below.
    def maxMin(k, arr):
        arr.sort()
        unfairness=-1
        for i in range(len(arr)-k+1):
            temp = arr[i+k-1]- arr[i]
            if unfairness==-1:
                unfairness=temp
            else:
                unfairness=min(unfairness,temp)
        return unfairness
        
    n = int(input())
    
    k = int(input())
    
    arr = []
    
    for _ in range(n):
        arr_item = int(input())
        arr.append(arr_item)
    
    result = maxMin(k, arr)
    print(str(result))
    
    return


def cloudy_day():
    def maximumPeople(p, x, y, r):
        # Return the maximum number of people that will be in a sunny town after removing exactly one cloud.
        
        def _sum_sunny(p,x,clouds):
            # for each cloud find cities made sunny by removing
            if len(clouds) == 0:
                return sum(p)
            else:
                return sum(p)-sum([p[i] for i in set([j for j, town in enumerate(x) for i in clouds if i[0] <= town <= i[1]])])
        
        
        x_simple = list(set(x))
        p_simple = [sum([p[j] for j, town_coord in enumerate(x) if x_coord==town_coord]) for x_coord in x_simple]
        
        # ord = (lambda s: sorted(range(1, len(s) + 1), key=lambda i: s[i - 1]))(x)
        # x_sort = [x[i - 1] for i in ord]
        # p_sort = [p[i - 1] for i in ord]
        clouds = [[y[i] - r[i], y[i] + r[i]] for i in range(len(r))]
        # ord_clouds = (lambda s: sorted(range(1, len(s) + 1), key=lambda i: s[i - 1][0]))(clouds)
        # clouds_sort = [clouds[i - 1] for i in (lambda s: sorted(range(1, len(s) + 1), key=lambda i: s[i - 1][0]))(clouds)]

        # find list of clouds that cover at least one town
        clouds_filter = [clouds[i] for i, cd in enumerate(clouds) for j in x_simple if cd[0] <= j <= cd[1]]
        # clouds_filter = [clouds_sort[i] for i, cd in enumerate(clouds_sort) for j in x_sort if cd[0] <= j <= cd[1]]
        
        if len(clouds_filter)==0:
            return sum(p_simple)
            # return sum(p_sort)
        else:
            # find list of towns that are not yet sunny
            d = set([j for j, town in enumerate(x_simple) for i in clouds_filter if i[0] <= town <= i[1]])
            # d = set([j for j, town in enumerate(x_sort) for i in clouds_filter if i[0] <= town <= i[1]])
            
            x_dark = [x_simple[i] for i in d]
            p_dark = [p_simple[i] for i in d]
            p_sunny = sum(p_simple) - sum(p_dark)
            # x_dark = [x_sort[i] for i in d]
            # p_dark = [p_sort[i] for i in d]
            # p_sunny = sum(p_sort) - sum(p_dark)

            return p_sunny+max([_sum_sunny(p_dark,x_dark,clouds_filter[:i]+clouds_filter[i+1:]) for i in range(len(clouds_filter))])

    n = int(input())

    p = list(map(int, input().rstrip().split()))

    x = list(map(int, input().rstrip().split()))

    m = int(input())

    y = list(map(int, input().rstrip().split()))

    r = list(map(int, input().rstrip().split()))

    result = maximumPeople(p, x, y, r)
    print(str(result))
    return


def largest_permutation():
    # Complete the largestPermutation function below.
    def largestPermutation(k, arr):
        l=[]
        arr.reverse()
        while (k>0) and (len(arr)>1):
            m = max(arr)
            o = arr.pop()
            l.append(m)
            if o!=m:
                i = arr.index(m)
                arr[i] = o
                k-=1
        arr.reverse()
        l.extend(arr)
        return l
    
    nk = input().split()
    
    n = int(nk[0])
    
    k = int(nk[1])
    
    arr = list(map(int, input().rstrip().split()))
    
    result = largestPermutation(k, arr)
    
    print(' '.join(map(str, result)))


def birthday_chocolate():
    import math
    import os
    import random
    import re
    import sys
    
    def birthday(s, d, m):
        if m>len(s):
            return 0
        else:
            return sum([sum(s[i:i+m])==d for i in range(len(s)+1-m)])

    n = int(input().strip())

    s = list(map(int, input().rstrip().split()))

    dm = input().rstrip().split()

    d = int(dm[0])

    m = int(dm[1])

    result = birthday(s, d, m)
    
    print(str(result))


def divisible_sum_pairs():
    import math
    import os
    import random
    import re
    import sys
    
    # Complete the divisibleSumPairs function below.
    def divisibleSumPairs(n, k, ar):
        return sum([((ar[i] + ar[j]) % k) == 0 for i in range(n - 1) for j in range(i + 1, n)])
        
    nk = input().split()

    n = int(nk[0])

    k = int(nk[1])

    ar = list(map(int, input().rstrip().split()))

    result = divisibleSumPairs(n, k, ar)
    
    print(str(result))


def migratory_birds():
    import math
    import os
    import random
    import re
    import sys

    def migratoryBirds(arr):
        arr.sort()
        max_c = 1
        c = 1
        x0 = arr[0]
        max_v = x0
        for x in arr[1:]:
            if x==x0:
                c+=1
                if c>max_c:
                    max_c = c
                    max_v = x
            else:
                x0=x
                c=1
        return max_v

    arr_count = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    result = migratoryBirds(arr)
    
    print(str(result))


def day_of_the_programmer():
    import math
    import os
    import random
    import re
    import sys
    
    # Complete the dayOfProgrammer function below.
    def dayOfProgrammer(year):
        if year == 1918:
            return "26.09.1918"
        if year % 4 == 0 and (year < 1918 or year % 400 == 0 or year % 100 != 0):
            return "12.09.%s" % year
        return "13.09.%s" % year

    year = int(input().strip())

    result = dayOfProgrammer(year)
    
    print(str(result))


def Encryption():
    import math
    import os
    import random
    import re
    import sys
    
    # Complete the encryption function below.
    def encryption(s):
        s = s.replace(" ", "")
        rt = math.sqrt(len(s))
        r_min = math.floor(rt)
        c_max = math.ceil(rt)
        if len(s) > (r_min * c_max):
            r_min += 1
    
        result = " ".join(
            ["".join([s[i * c_max + j] if (i * c_max + j) < len(s) else "" for i in range(r_min)]) for j in
             range(c_max)])
    
        return result

    s = input()

    result = encryption(s)
    
    print(str(result))


def bigger_is_greater():
    import math
    import os
    import random
    import re
    import sys
    
    # Complete the biggerIsGreater function below.
    def biggerIsGreater(w):
    
        l = [s for s in w]
        r = l.copy()
        r.reverse()
        s = l.copy()
        s.sort()
        if s == r:
            return 'no answer'
    
        greater = False
        i = len(l) - 1
        while not greater:
            if l[i - 1] < l[i]:
                pivot = l[i-1]
                end = l[i:]
                for j in range(len(end)):
                    if end[-(j+1)]>pivot:
                        swap = end.pop(len(end)-(j+1))
                        break
                end.append(pivot)
                end.sort()
                res = l[:i - 1] + [swap] + end
                greater = True
            else:
                i -= 1
    
        return "".join(res)

    T = int(input())

    for T_itr in range(T):
        w = input()
    
        result = biggerIsGreater(w)
        
        print(result)


def modifed_kaprekar():
    import math
    import os
    import random
    import re
    import sys
    
    # Complete the kaprekarNumbers function below.
    def kaprekarNumbers(p, q):
        res = [str(r) for r in (n if sum(map(int, [str(n ** 2)[:int(len(str(n ** 2)) / 2)], str(n ** 2)[int(len(str(n ** 2)) / 2):]] if len(str(n ** 2))>1 else [str(n ** 2)])) == n else None for n in range(p,q+1)) if r is not None]
        return " ".join(res) if len(res)>0 else "INVALID RANGE"

    p = int(input())

    q = int(input())

    print(kaprekarNumbers(p, q))


def emas_supercomputer():
    import math
    import os
    import random
    import re
    import sys
    
    # Complete the twoPluses function below.
    def twoPluses(grid):
    
        def _common_member(a, b):
            a_set = set(a)
            b_set = set(b)
            if (a_set & b_set):
                return True
            else:
                return False
        
        def _get_radius(c,r,grid):
            if grid[r][c] == "B":
                yield (0, None)
            else:
                ret_l = [(r,c)]
                yield (1, ret_l.copy())
                col = [i[c] for i in grid]
                row = grid[r]
                n = r-1
                s = r+1
                e = c+1
                w = c-1
                length = 3
                cont = True
                while cont and (0<=n<len(grid)) and (0<=s<len(grid)) and (0<=e<len(grid[0])) and (0<=w<len(grid[0])):
                    if grid[n][c] == grid[s][c] == grid[r][e] == grid[r][w] == "G":
                        ret_l.extend([(n,c),(s,c),(r,e),(r,w)])
                        yield (2*length-1, ret_l.copy())
                    else:
                        cont = False
                    length+=2
                    n-=1
                    s+=1
                    e+=1
                    w-=1
                

        all_rads = [list(_get_radius(c, r, grid)) for r in range(len(grid)) for c in range(len(grid[0]))]
        all_rads_squash = [item for sublist in all_rads for item in sublist]
        all_rads_clean = [x for x in all_rads_squash if x[0] > 0]
        all_rads_clean.sort(reverse=True)

        max_c = 0
        for i, (v0, l0) in enumerate(all_rads_clean[:-1]):
            for j1, (v1, l1) in enumerate(all_rads_clean[i + 1:]):
                if v0 * v1 > max_c:
                    if not _common_member(l0, l1):
                        max_c = v0 * v1
        
        return max_c

    with open('input01.txt', 'r') as fptr:
    
        nm = fptr.readline().rstrip().split()

        n = int(nm[0])
    
        m = int(nm[1])

        grid = []

        for _ in range(n):
            grid_item = fptr.readline().rstrip()
            grid.append(grid_item)

    result = twoPluses(grid)
    
    print(str(result))


def larrys_array():
    import math
    import os
    import random
    import re
    import sys
    
    def _final_three(A, A_sorted):
        if A==A_sorted:
            return "YES"
        for a in range(3):
            A = _rotate(A, len(A)-3)
            if A == A_sorted:
                return "YES"
        return "NO"
    
    def _rotate(A, sort_ix):
        # s = A[sort_ix:sort_ix+3]
        # A[sort_ix:sort_ix + 3] = s[1:]+[s[0]]
        A[sort_ix:sort_ix + 3] = A[sort_ix+1:sort_ix+3]+[A[sort_ix]]
        return A
    
    # Complete the larrysArray function below.
    def larrysArray(A):
        A_sorted = A.copy()
        A_sorted.sort()
        
        for i in range(1,len(A)-1):
            if i==(len(A)-2):
                return _final_three(A, A_sorted)
            else:
                if A == A_sorted:
                    return "YES"
                ix = A.index(i)
                while ix > (i-1):
                    sort_ix = max([i-1,ix-2])
                    A = _rotate(A, sort_ix)
                    ix = A.index(i)

    with open('input01.txt', 'r') as fptr:
        
        t = int(fptr.readline().rstrip())
    
        for t_itr in range(t):
            n = int(fptr.readline().rstrip())
            
            A = list(map(int, fptr.readline().rstrip().split()))

            result = larrysArray(A)

            print(str(result))
    
    return


def zig_zag_sequence():
    def findZigZagSequence(a, n):
        a.sort()
        mid = int((n - 1) / 2)
        a[mid], a[mid+1:] = a[-1], a[mid:-1]
        
        st = mid + 1
        ed = n - 1
        while (st <= ed):
            a[st], a[ed] = a[ed], a[st]
            st = st + 1
            ed = ed - 1
        
        for i in range(n):
            if i == n - 1:
                print(a[i])
            else:
                print(a[i], end=' ')
        return

    test_cases = int(input())
    for cs in range(test_cases):
        n = int(input())
        a = list(map(int, input().split()))
        findZigZagSequence(a, n)
    
    return


def sparse_arrays():
    def matchingStrings(strings, queries):
        import collections
        
        values = collections.defaultdict(int)
        for s in strings:
            values[s] += 1
        
        return [values[q] for q in queries]

    strings_count = int(input())

    strings = []

    for _ in range(strings_count):
        strings_item = input()
        strings.append(strings_item)

    queries_count = int(input())

    queries = []

    for _ in range(queries_count):
        queries_item = input()
        queries.append(queries_item)

    res = matchingStrings(strings, queries)
    
    for r in res:
        print(str(r))


def array_manipulation():
    def arrayManipulation(n, queries):
        
        arr = [0]*(n+1)
        for q in queries:
            a,b,k = q
            arr[a - 1] += k
            arr[b] -= k
        
        sm = 0
        max_val = 0
        for x in arr:
            sm += x
            max_val = sm if sm > max_val else max_val
            
        return max_val

    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    queries = []

    for _ in range(m):
        queries.append(list(map(int, input().rstrip().split())))

    result = arrayManipulation(n, queries)
    
    print(str(result))


def main():
    array_manipulation()
    return


if __name__ == '__main__':
    main()