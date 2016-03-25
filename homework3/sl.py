# CS 181, Harvard University
# Spring 2016
import numpy as np 
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron
import csv
from sklearn import cross_validation
import time
import os

# Implement this class
def basicKernel(x, x1):
    return np.dot(x, x1)

def nontrivial_kernel(x1,x2):
    return np.exp(-1 * np.power(np.linalg.norm(x1 - x2), 2))

def stupid(x1,x2):
    return nontrivial_kernel(x1,x2) * basicKernel(x1,x2)

class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples

    def __shuffle(self, X, Y):
        join = np.concatenate((X, np.array([Y]).T), axis=1)
        np.random.shuffle(join)
        shuffled = np.hsplit(join, np.array([2, 2]))
        self.X = shuffled[0]
        self.Y = np.ndarray.flatten(shuffled[2])
        self.alpha = np.zeros(Y.shape, dtype=np.float64)

    # Implement this!
    def fit(self, X, Y):
        self.support = np.empty(0)
        self.__shuffle(X, Y)

        for t in range(numsamples):
            yhat = 0
            for i in self.support:
                yhat += self.alpha[i] * stupid(self.X[t], self.X[i])
            if self.Y[t] * yhat <= 0:
                self.support = np.append(self.support, t)
                self.alpha[t] = self.Y[t]

    # Implement this!
    def predict(self, X):
        predict = np.empty(0)
        for x in X:
            res = 0
            for i in self.support:
                res += self.alpha[i] * stupid(x, self.X[i])
            if res == 0:
                predict = np.append(predict, -1)
            else:
                predict = np.append(predict, np.sign(res))
        return predict

# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples

    def __shuffle(self, X, Y):
        join = np.concatenate((X, np.array([Y]).T), axis=1)
        np.random.shuffle(join)
        shuffled = np.hsplit(join, np.array([2, 2]))
        self.X = shuffled[0]
        self.Y = np.ndarray.flatten(shuffled[2])
        self.alpha = np.zeros(Y.shape, dtype=np.float64)

    # Implement this!
    def fit(self, X, Y):
        self.support = np.empty(0)
        self.yhats = np.empty(0)
        self.__shuffle(X, Y)

        def format(i):
            return self.Y[i] * (self.yhats[i] - self.alpha[i] * stupid(self.X[i], self.X[i]))

        vformat = np.vectorize(format)

        for t in range(numsamples):
            yhat = 0
            for i in self.support:
                yhat += self.alpha[i] * stupid(self.X[t], self.X[i])
            self.yhats = np.append(self.yhats, yhat)
            if self.Y[t] * yhat <= self.beta:
                self.support = np.append(self.support, t)
                self.alpha[t] = self.Y[t]
                if len(self.support) > self.N:
                    maxarray = vformat(self.support)
                    self.support = np.delete(self.support, np.argmax(maxarray))
            if not t % 1000:
                print t

    # Implement this!
    def predict(self, X):
        predict = np.empty(0)
        for x in X:
            res = 0
            for i in self.support:
                res += self.alpha[i] * stupid(x, self.X[i])
            if res == 0:
                predict = np.append(predict, -1)
            else:
                predict = np.append(predict, np.sign(res))
        return predict

class SMO(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples
        # unsure about this
        self.C = 100
        self.B = np.empty(0, dtype=int)
        self.A = np.empty(0, dtype=int)
        self.tau = 0.00001
        self.iterations = 100

    def __isViolated(self, i, j):
        return self.alpha[i] < self.B[i] or self.alpha[j] > self.A[j] or self.gradient[i] - self.gradient[j] < self.tau

    def __update(self, i, j):
        print i, j
        grads = float(self.gradient[i] - self.gradient[j]) / (stupid(self.X[i], self.X[i]) + stupid(self.X[j], self.X[j]) - 2 * stupid(self.X[i], self.X[j]))
        lam = min(grads, self.B[i] - self.alpha[i], self.alpha[j] - self.A[j])
        self.alpha[i] += lam
        self.alpha[j] -= lam
        for s in range(self.numsamples):
            self.gradient[s] -= lam * (stupid(self.X[i], self.X[s]) - stupid(self.X[j], self.X[s]))

    def __getGradients(self):
        for k in range(self.numsamples):
            if not k % 500:
                print k
            yhat = 0
            for i in range(self.numsamples):
                yhat += self.alpha[i] * stupid(self.X[i], self.X[k])
            self.gradient = np.append(self.gradient, Y[k] - yhat)

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        for y in self.Y:
            self.A = np.append(self.A, min(0, self.C * y))
            self.B = np.append(self.B, max(0, self.C * y))
        self.alpha = np.zeros(Y.shape, dtype=np.float64)
        self.gradient = np.empty(0)
        self.__getGradients()

        for n in range(self.iterations):
            if not n % 10:
                print "here we go: " + str(n)
            found = False
            for i in range(self.numsamples):
                for j in range(self.numsamples):
                    if not found and i != j and self.gradient[i] != self.gradient[j] and self.__isViolated(i, j):
                        found = True
                        self.__update(i, j)
            if not found:
                print "no violating pairs found"
                break

    def predict(self, X):
        predict = np.empty(0)
        for x in X:
            res = 0
            for i in range(self.numsamples):
                res += self.alpha[i] * stupid(x, self.X[i])
            if res == 0:
                predict = np.append(predict, -1)
            else:
                predict = np.append(predict, np.sign(res))
        return predict

class LASVM(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.numsamples = numsamples
        self.lowerb = 0
        self.delta = 0
        self.C = 100000
        self.B = np.empty(0, dtype=int)
        self.A = np.empty(0, dtype=int)
        self.tau = 0.00001
        self.iterations = 2000

    def __shuffle(self, X, Y):
        join = np.concatenate((X, np.array([Y]).T), axis=1)
        np.random.shuffle(join)
        shuffled = np.hsplit(join, np.array([2, 2]))
        self.X = shuffled[0]
        self.Y = np.ndarray.flatten(shuffled[2])
        self.alpha = np.zeros(Y.shape, dtype=np.float64)

    def __isViolated(self, i, j):
        return self.alpha[i] < self.B[i] and self.alpha[j] > self.A[j] and self.gradient[i] - self.gradient[j] > self.tau

    def __update(self, i, j):
        if i != j:
            # print i, j
            grads = float(self.gradient[i] - self.gradient[j]) / (stupid(self.X[i], self.X[i]) + stupid(self.X[j], self.X[j]) - 2 * stupid(self.X[i], self.X[j]))
            lam = min(grads, self.B[i] - self.alpha[i], self.alpha[j] - self.A[j])
            self.alpha[i] += lam
            self.alpha[j] -= lam
            for s in self.support:
                self.gradient[s] -= lam * (stupid(self.X[i], self.X[s]) - stupid(self.X[j], self.X[s]))

    def __process(self, k):
        if k in self.support:
            return -1
        self.alpha[k] = 0
        tmp = 0
        for s in self.support:
            tmp += self.alpha[s] * stupid(self.X[k], self.X[s])
        self.gradient[k] = self.Y[k] - tmp
        self.support = np.append(self.support, k)
        if self.Y[k] == 1:
            i = k
            j = self.support[np.argmin(self.gradient[self.support[np.where(self.alpha[self.support] > self.A[self.support])]])]
        else:
            j = k
            i = self.support[np.argmax(self.gradient[self.support[np.where(self.alpha[self.support] < self.B[self.support])]])]
        if not self.__isViolated(i, j):
            return -1
        self.__update(i, j)

    def __reprocess(self):
        i = self.support[np.argmax(self.gradient[self.support[np.where(self.alpha[self.support] < self.B[self.support])]])]
        j = self.support[np.argmin(self.gradient[self.support[np.where(self.alpha[self.support] > self.A[self.support])]])]
        if not self.__isViolated(i, j):
            return -1
        self.__update(i, j)
        i = self.support[np.argmax(self.gradient[self.support[np.where(self.alpha[self.support] < self.B[self.support])]])]
        j = self.support[np.argmin(self.gradient[self.support[np.where(self.alpha[self.support] > self.A[self.support])]])]
        for i, s in enumerate(self.support):
            if not self.alpha[s]:
                if self.Y[s] == -1 and self.gradient[s] >= self.gradient[i]:
                    np.delete(self.support, i)
                elif self.Y[s] == 1 and self.gradient[s] <= self.gradient[j]:
                    np.delete(self.support, i)
        self.lowerb = float(self.gradient[i] + self.gradient[j])/2
        self.delta = self.gradient[i] - self.gradient[j]

    def __seed(self):
        ind = 1
        self.support = np.append(self.support, 0)
        while list(self.Y[self.support]).count(1) < 40 or list(self.Y[self.support]).count(-1) < 40:
            self.support = np.append(self.support, ind)
            ind += 1

    def __getGradients(self):
        for k in range(self.numsamples):
            yhat = 0
            for i in range(self.numsamples):
                yhat += self.alpha[i] * stupid(self.X[i], self.X[k])
            self.gradient = np.append(self.gradient, Y[k] - yhat)

    def fit(self, X, Y):
        # self.__shuffle(X,Y)
        self.X = X
        self.Y = Y
        self.alpha = np.zeros(Y.shape, dtype=np.float64)
        for y in self.Y:
            self.A = np.append(self.A, min(0, self.C * y))
            self.B = np.append(self.B, max(0, self.C * y))
        self.support = np.empty(0, dtype=int)
        self.gradient = np.empty(0)
        self.__seed()
        self.__getGradients()
        for n in range(self.iterations):
            # print "iteration: ", n
            self.__process(random.randint(0, self.numsamples-1))
            self.__reprocess()
        while self.delta >= self.tau:
            # print self.delta
            # print self.tau
            if self.__reprocess() == -1:
                # print "RUN AWAY"
                break

    def predict(self, X):
        predict = np.empty(0)
        for i,x in enumerate(X):
            # if not i % 1000:
            #     print i
            res = 0
            for i in self.support:
                res += self.alpha[i] * stupid(x, self.X[i])
            if res == 0:
                predict = np.append(predict, -1)
            else:
                predict = np.append(predict, np.sign(res))
        return predict

# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 100
# numsamples = 20000
numsamples = 2000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'
smo_file_name = 'smo.png'
lasvm_file_name = 'lasvm.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
# k = KernelPerceptron(numsamples)
# k.fit(X,Y)
# k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

# bk = BudgetKernelPerceptron(beta, N, numsamples)
# bk.fit(X, Y)
# bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

# smo = SMO(beta, N, numsamples)
# smo.fit(X,Y)
# smo.visualize(smo_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

os.chdir('./lasvm_time')
lasvm = LASVM(beta, N, numsamples)
# lasvm.fit(X,Y)
# lasvm.visualize(lasvm_file_name, width=0, show_charts=True, save_fig=True, include_points=False)


# timing script 
# iterations = 1
# fname = 'lasvm_'+str(beta)+'_'+str(N) + '_' + str(numsamples)+'_'+'first_timing_stoopid.csv'

# with open(fname, 'wb') as outcsv:
#     writer = csv.writer(outcsv)
#     writer.writerow(['iteration', 'seconds'])
#     outcsv.close()

# with open(fname, 'a') as fl:
#     writer = csv.writer(fl)
#     for i in range(1, iterations + 1):
#         print i 
#         start = time.time()
#         lasvm.fit(X,Y)
#         end = time.time() - start
#         writer.writerow([i, end])
#         print "Time: " + str(end)

# # accuracy script 
# iterations = 5 
# fname = 'lasvm_'+str(beta)+'_'+str(N) + '_' + str(numsamples)+'_'+str(iterations)+'_accuracy_linear.csv'
# with open(fname, 'wb') as outcsv:
#     writer = csv.writer(outcsv)
#     writer.writerow(['iteration', 'accuracy'])
#     outcsv.close()

# with open(fname, 'a') as fl:
#     writer = csv.writer(fl)
#     for i in range(1,iterations + 1):
#         print i
#         join = np.concatenate((X, np.array([Y]).T), axis=1)
#         np.random.shuffle(join)
#         shuffled = np.hsplit(join, np.array([2, 2]))
#         X = shuffled[0]
#         Y = np.ndarray.flatten(shuffled[2])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)
print "STARTING THIS STEEEEEP"
lasvm.fit(X_train,y_train)
print "DOOOOONE WITH THIS STEEEEEP"
predict = lasvm.predict(X_test)
print "LET'S GET CHECKIIING"
count = 0 
for j in range(y_test.shape[0]):
    if predict[j] == y_test[j]:
        count += 1
print float(count)/y_test.shape[0]
