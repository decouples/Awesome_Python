# 实现一个累加
import time
import timeit
def sumofN(n):
    start = time.time()
    theSum = 0
    for i in range(1,n+1):
        theSum = theSum+i
    end = time.time()
    return theSum,end-start

def sumofN2(n):
    start = time.time()
    theSum = 0;
    for i in range(1,n+1):
        theSum = theSum+i
    end = time.time()
    return theSum,end-start

def sumofN3(n):
    start = time.time()
    end = time.time()
    return (n*(n+1))/2,end-start

# give a test the running time
# for i in range(5):
#     print("sumofN :Sum is %d required %10.7f seconds "%sumofN(10000000))
# for i in range(5):
#     print("sumofN2 :Sum is %d required %10.7f seconds "%sumofN2(10000000))
# for i in range(5):
#     print("sumofN3 :Sum is %d required %10.7f seconds "%sumofN3(10000000))
# 以下实现了一个两个字符串判定是否是乱序
def anagramSolution4(s1,s2):
    c1=[0]*26
    c2=[0]*26
    for i in range(len(s1)):
        pos = ord(s1[i])-ord('a')
        c1[pos] = c1[pos]+1
    for i in range(len(s2)):
        pos = ord(s2[i])-ord('a')
        c2[pos]=c2[pos]+1
    j=0
    stillOK = True
    while j<26 and stillOK:
        if c1[j] == c2[j]:
            j=j+1
        else:
            stillOK=False
    return stillOK

# print(anagramSolution4('apple','pleap'))
#
# 关于列表的一些操作
def test1():
    l =[]
    for i in range(1000):
        l = l+[i]
def test2():
    l=[]
    for i in range(1000):
        l.append(i)
def test3():
    l = [i for i in range(1000)]
def test4():
    l = list(range(1000))

# t1 = Timer("test1()","from __main__ import test1")
# print("concat ",t1.timeit(number=1000),"milliseconds")
#
# t1 = time("test2()","from __main__ import test1")
# print("append ",t1.timeit(number=1000),"milliseconds")
#
# t1 = time("test3()","from __main__ import test1")
# print("comprehension ",t1.timeit(number=1000),"milliseconds")
#
# t1 = time("test4()","from __main__ import test1")
# print("list range ",t1.timeit(number=1000),"milliseconds")

# popzero = timeit.Timer("x.pop(0)","from __main__ import x")
# popend = timeit.Timer("x.pop(0)","from __main__ import x")
#
# x = list(range(2000000))
# popzero.timeit(number=1000)
#
# x = list(range(2000000))
# popzero.timeit(number=1000)

# import timeit
# import random
#
# for i in range(10000,1000001,20000):
#     t = timeit.Timer("random.randrange(%d) in x"%i,"from __main__ import random,x")
#     x = list(range(i))
#     list_time = t.timeit(number=1000)
#     x = {j:None for j in range(i)}
#     d_time = t.timeit(number=1000)
#     print("%d,%10.3f,%10.3f"%(i,list_time,d_time))

# x = {j:None for j in range(10000)}
# y = [i for i in range(10000)]
# z = list(range(10000))
# print(x)
# print(y)
# print(z)

# 一些关于栈的知识
class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

# s = Stack()
# print(s.isEmpty())
# s.push(4)
# s.push('dog')
# print(s.peek())
# s.push(True)
# print(s.size())
# print(s.isEmpty())
# s.push(8.4)
# print(s.pop())
# print(s.pop())
# print(s.size())
#
# 一个简单的括号检测函数，使用栈来实现
def parChecker(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol == "(":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                s.pop()
        index =index+1
    if balanced and s.isEmpty():
        return True
    else:
        return False

# print(parChecker('((()))'))
# print(parChecker('(()'))

def parChecker1(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol == "(":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                s.pop()
                if not matches(top,symbol):
                    balanced = False
        index = index+1
        index =index+1
    if balanced and s.isEmpty():
        return True
    else:
        return False
def matches(open,close):
    opens = "([{"
    closers = ")]}"
    return opens.index(open) == closers.index(close)

# print(parChecker1('{{([][])}()}'))
# print(parChecker1('[]'))

# 二进制字符串
def divideBy2(decNumber):
    remstack = Stack()
    while decNumber > 0:
        rem = decNumber % 2
        remstack.push(rem)
        decNumber = decNumber // 2
    binString = ""
    while not remstack.isEmpty():
        binString = binString + str(remstack.pop())
    return binString

# print(divideBy2(42))
# 定义一个通用的进制转换
def baseConvert(decNumbert,base):
    digits = "0123456789ABCDEF"
    remstack = Stack()
    while decNumbert > 0:
        rem = decNumbert % base
        remstack.push(rem)
        decNumbert = decNumbert // base
    newString = ""
    while not remstack.isEmpty():
        newString = newString + digits[remstack.pop()]
    return newString


# print(baseConvert(135, 2))
# print(baseConvert(13, 16))

# 以下编写关于队列的一些知识

class Queue:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def enqueue(self,item):
        self.items.insert(0,item)
    def dequeue(self):
        return self.items.pop()
    def size(self):
        return len(self.items)

# s = Queue()
# print(s.isEmpty())
# s.enqueue('dog')
# s.enqueue(84)
# print(s.size())
# s.dequeue()
# print(s.size())

# 模拟一个队列的操作
def hotPotato(namelist,num):
    simqueue = Queue()
    for name in namelist:
        simqueue.enqueue(name)
    while simqueue.size() > 1:
        for i in range(num):
            simqueue.enqueue(simqueue.dequeue())
        print(simqueue.dequeue())
        # 显示每一次删除的是谁
    return simqueue.dequeue()

# print(hotPotato(["Bill","David","Susan","Jack","Kent","Brad"],3))

# 演示一个打印序列
class Printer:
    def __init__(self,ppm):
        self.pagerate = ppm
        self.currentTask = None
        self.timeRemaining = 0

    def tick(self):
        if self.currentTask != None:
            self.timeRemaining = self.timeRemaining - 1
            if self.timeRemaining <= 0:
                self.currentTask = None

    def busy(self):
        if self.currentTask != None:
            return True
        else:
            return False

    def startNext(self,newtask):
        self.currentTask = newtask
        self.timeRemaining = newtask.getPages()*60/self.pagerate


import random


class Task:
    def __init__(self,time):
        self.timestamp = time
        self.pages = random.randrange(1,21)

    def getStamp(self):
        return self.timestamp

    def getPages(self):
        return self.pages

    def waitTime(self,currenttime):
        return currenttime - self.timestamp

def simulation(numSeconds,pagesPerMinute):
    labprinter = Printer(pagesPerMinute)
    printQueue = Queue()
    waitingtimes = []
    for curentSecond in range(numSeconds):
        if newPrintTask():
            task =Task(curentSecond)
            printQueue.enqueue(task)
        if (not labprinter.busy()) and (not printQueue.isEmpty()):
            nexttask = printQueue.dequeue()
            waitingtimes.append(nexttask.waitTime(curentSecond))
            labprinter.startNext(nexttask)
        labprinter.tick()
    averageWait = sum(waitingtimes)/len(waitingtimes)
    print("Average wait %6.2f secs %3d tasks remaining. "%(averageWait,printQueue.size()))
def newPrintTask():
    num = random.randrange(1,181)
    if num ==180:
        return True
    else:
        return False


# for i in range(10):
#     simulation(3600,10)

# 双端队列的实现
class Deque:
    def __init__(self):
        self.items = []

    def isEmpty(self,item):
        self.items.append(item)

    def addRear(self,item):
        self.items.insert(0,item)

    def removeFront(self):
        return self.items.pop()

    def removeReare(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)

# 利用双端队列检查回文


def palchecker(aString):
    chardeque = Deque()
    for ch in aString:
        chardeque.addRear(ch)
    stillEqual = True
    while chardeque.size() > 1 and stillEqual:
        first = chardeque.removeFront()
        last = chardeque.removeReare()
        if first != last:
            stillEqual = False
    return stillEqual

# print(palchecker("lsdkjfskf"))
# print(palchecker("raar"))

# 定义一个列表


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    def get_data(self):
        return self.data

    def get_next(self):
        return self.next

    def set_data(self, new_data):
        self.data = new_data

    def set_next(self, new_next):
        self.next = new_next




# temp = Node(93)
# print(temp.get_data())


class UnorderedList:
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head == None

    def add(self,item):
        temp = Node(item)
        temp.set_next(self.head)

    def size(self):
        current = self.head
        count = 0
        while current != None:
            count = count+1
            current = current.get_next()
        return count

    def search(self,item):
        current = self.head
        found = False
        while current != None and not found:
            if current.get_data() == item:
                found = True
            else:
                current = current.get_next()
        return found

    def remove(self,item):
        current = self.head
        previous = None
        found = False
        while not found:
            if current.get_data() == item:
                found = True
            else:
                previous = current
                current = current.get_next
        if previous == None:
            self.head = current.get_next()
        else:
            previous.set_next(current.get_next())


# my_list = UnorderedList()
# print(my_list.is_empty())


class OrderedList:
    def __init__(self):
        self.head = None

    def search(self,item):
        current = self.head
        found = False
        stop = False
        while current != None and not stop:
            if current.get_data() == item:
                founf = True
            else:
                if current.get_data() > item:
                    stop = True
                else:
                    current = current.get_next()
        return found

    def add(self,item):
        current = self.head
        previous = None
        stop = False
        while current != None and not stop:
            if current.get_data() > item:
                stop = True
            else:
                previous = current
                current = current.get_next
        temp = Node(item)
        if previous == None:
            temp.set_next(self.head)
            self.head = temp
        else:
            temp.set_next(current)
            previous.set_next(temp)
