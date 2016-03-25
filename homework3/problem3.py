# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron
import sys
import time
import csv
import os 
from sklearn import cross_validation

###################################################################################
# HELPER FUNCTIONS 

# Linear kernel 
def trivial_kernel(x1, x2):
   	return np.dot(x1, x2)

# Gaussian/RBF kernel
def nontrivial_kernel(x1,x2):
	temp = -1 * np.power(np.linalg.norm(x1-x2),2)
	return np.exp(temp)

# Product of Gaussian and Linear kernel
# Proof of this being a valid kernel: 
# Refer to problem 2 on the pset 
def stupid_kernel(x1,x2):
	return trivial_kernel(x1,x2) * nontrivial_kernel(x1,x2)

# Shuffles the X and Y values to ensure that 
# we always take random exmamples
def shuffle(X, Y):
	# Necessary to ensure that Y values are correctly
	# associated with the respective X values
	z = np.concatenate((X,np.array([Y]).T), axis = 1)

	# Randomizes everything 
	np.random.shuffle(z)

	# Splits them back up into two arrays
	new_vals = np.hsplit(z, np.array([2,2]))
	new_X = new_vals[0]
	new_Y = np.ndarray.flatten(new_vals[2])

	# Update
	X = new_X
	Y = new_Y
	return X,Y

###################################################################################

# Implement this class
class KernelPerceptron(Perceptron):
	def __init__(self, numsamples):
		# this is 20000 (for now)
		self.numsamples = numsamples

	# Implement this!
	def fit(self, X, Y):
		# Instantiates the support as an empty array
		self.support = np.empty(0)
		self.X = X
		self.Y = Y
		
		# Instantiates the alpha values as an array of zeroes 
		self.alpha = np.zeros(Y.shape, dtype = np.float64)

		# Iterate over however many samples we specify
		for t in range(numsamples):
			sample = self.X[t]
			y_hat = 0
			# Compute y_hat, where b = 0 
			for i in self.support:
				y_hat += self.alpha[i] * trivial_kernel(self.X[t], self.X[i])
			if self.Y[t] * y_hat <= 0:
				self.support = np.append(self.support, t)
				self.alpha[t] = self.Y[t]
		
	# Implement this!
	def predict(self, X):
		# Instantiate an empty array for predictions
		prediction = np.empty(0)
		for x in X: 
			# best_guess is basically y_hat
			best_guess = 0
			for j in self.support:
				best_guess += self.alpha[j] * trivial_kernel(x, self.X[j])
			# Necessary because the first value in the alphas
			# array will always be 0 
			if best_guess == 0:
				prediction = np.append(prediction, -1)
			# Check sign of y_hat and assign to respective class
			else:
				prediction = np.append(prediction, np.sign(best_guess))	
		return prediction	

	# Returns length of self.support, which is the number of support vectors 
	def sv(self):
		return len(self.support)	

###################################################################################

# Implement this class
class BudgetKernelPerceptron(Perceptron):
	def __init__(self, beta, N, numsamples):
		self.beta = beta
		self.N = N
		self.numsamples = numsamples
		
	# Implement this!
	def fit(self, X, Y):
		# Instantiates support as an empty array
		self.support = np.empty(0)
		self.X = X
		self.Y = Y
		# Necessary to store y_hat values for the update phase when we 
		# remove vectors from the support if its size is greater than N
		self.y_hat_vals = np.empty(0)
		# Instantiates the alpha values as an array of zeroes 
		self.alpha = np.zeros(Y.shape, dtype = np.float64)

		# Poorly named function that handles the calculation in step 4b 
		def whatever(x):
			return self.Y[x] * (self.y_hat_vals[x] - self.alpha[x] *trivial_kernel(self.X[x], self.X[x]))

		# Create a higher order function vfunc that will be applied over the support
		vfunc = np.vectorize(whatever)

		# Basically the same as the fit in KernelPerceptron
		for t in range(numsamples):
			y_hat = 0
			for i in self.support:
				y_hat += self.alpha[i] * trivial_kernel(self.X[t], self.X[i])
			# Store y_hat values for calculation in whatever
			self.y_hat_vals = np.append(self.y_hat_vals, y_hat)
			if self.Y[t] * y_hat <= beta:
				self.support = np.append(self.support, t)
				self.alpha[t] = self.Y[t]
				# Check to see if cardinality of support is greater than maximum
				# number of support vectors allowed
				if len(self.support) > N:
					# Apply the whatever function over the support
					array = vfunc(self.support)
					# Find the index of the maximum element in the array and 
					# remove that element by its index
					self.support = np.delete(self.support, np.argmax(array))


	# Implement this!
	# Same as the predict for KernelPerceptron
	def predict(self, X):
		prediction = np.empty(0)
		for x in X: 
			best_guess = 0
			for j in self.support:
				best_guess += self.alpha[j] * trivial_kernel(x, self.X[j])
			if best_guess == 0:
				prediction = np.append(prediction, -1)
			else:
				prediction = np.append(prediction, np.sign(best_guess))
		return prediction

###################################################################################
#																				  #
#							EXTRA CREDIT IMPLEMENTATIONS 						  #
#																				  #
###################################################################################

class SMO(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples
        # In the paper, it says to set C to a 'reasonably high' value. 100 seems pretty
        # big IMO
        self.C = 100
        self.B = np.empty(0, dtype=int)
        self.A = np.empty(0, dtype=int)
        # After running it multiple times, this value of tau gave the best accuracy 
        self.tau = 0.00001
        # Predetermined number of iterations
        self.iterations = 1000

    # Checks to see if given example is a tau-violating pair
    def __tauViolation(self, i, j):
        return self.alpha[i] < self.B[i] or self.alpha[j] > self.A[j] or self.gradient[i] - self.gradient[j] < self.tau

    # Does all the updating nonsense 
    def __update(self, i, j):
    	# BASH BASH BASH
        gradients = float(self.gradient[i] - self.gradient[j]) / (stupid_kernel(self.X[i], self.X[i]) + stupid_kernel(self.X[j], self.X[j]) - 2 * stupid_kernel(self.X[i], self.X[j]))
        # Updates lambda
        l = min(gradients, self.B[i] - self.alpha[i], self.alpha[j] - self.A[j])
        # Updates alphas
        self.alpha[i] += l
        self.alpha[j] -= l
        # Update gradient for all values in NUMSAMPLES
        for x in range(self.numsamples):
            self.gradient[x] -= l * (stupid_kernel(self.X[i], self.X[x]) - stupid_kernel(self.X[j], self.X[x]))

    # Calcuates the gradient according to equation 9
    def __calcGrad(self):
        for k in range(self.numsamples):
            y_hat = 0
            for i in range(self.numsamples):
                y_hat += self.alpha[i] * stupid_kernel(self.X[i], self.X[k])
            self.gradient = np.append(self.gradient, Y[k] - y_hat)

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        # Calculates A and B according to equation 5
        for y in self.Y:
            self.A = np.append(self.A, min(0, self.C * y))
            self.B = np.append(self.B, max(0, self.C * y))
        # Same instantiation process as with the other two algorithms
        self.alpha = np.zeros(Y.shape, dtype=np.float64)
        self.gradient = np.empty(0)
        self.__calcGrad()

        # Iterates until we've iterated the predetermiend 
        # number of times or we can't find any more 
        # tau-violating pairs
        for n in range(self.iterations):
            found = False
            for i in range(self.numsamples):
                for j in range(self.numsamples):
                    if not found and i != j and self.gradient[i] != self.gradient[j] and self.__tauViolation(i, j):
                        found = True
                        self.__update(i, j)
            if not found:
                break

    # Same predict as before lol 
    def predict(self, X):
        predict = np.empty(0)
        for x in X:
            res = 0
            for i in range(self.numsamples):
                res += self.alpha[i] * stupid_kernel(x, self.X[i])
            if res == 0:
                predict = np.append(predict, -1)
            else:
                predict = np.append(predict, np.sign(res))
        return predict

###################################################################################

class LASVM(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.numsamples = numsamples
        # Bias and delta start at 0
        self.bias = 0
        self.delta = 0
        # "Reasonably large" value for C
        self.C = 100000
        self.B = np.empty(0, dtype=int)
        self.A = np.empty(0, dtype=int)
        self.tau = 0.00001
        self.iterations = 2000

    def __tauViolation(self, i, j):
        return self.alpha[i] < self.B[i] or self.alpha[j] > self.A[j] or self.gradient[i] - self.gradient[j] < self.tau

    # Does the updating nonsense but has the additional condition that 
    # the two indices cannot equal each other, otherwise we would have a value 
    # of 0, which would mess up gradient calculation 
    def __update(self, i, j):
        if i != j:
             gradients = float(self.gradient[i] - self.gradient[j]) / (stupid_kernel(self.X[i], self.X[i]) + stupid_kernel(self.X[j], self.X[j]) - 2 * stupid_kernel(self.X[i], self.X[j]))
	        # Updates lambda
	        l = min(gradients, self.B[i] - self.alpha[i], self.alpha[j] - self.A[j])
	        # Updates alphas
	        self.alpha[i] += l
	        self.alpha[j] -= l
	        # Update gradient for all values in NUMSAMPLES
	        for x in range(self.numsamples):
	            self.gradient[x] -= l * (stupid_kernel(self.X[i], self.X[x]) - stupid_kernel(self.X[j], self.X[x]))

	# LASVM Process method 
    def __process(self, k):
    	# BAIL OUT
        if k in self.support:
            return -1
        # Start this at 0
        self.alpha[k] = 0
        y_hat = 0
        for i in self.support:
            y_hat += self.alpha[i] * stupid_kernel(self.X[k], self.X[i])
        # Update gradient
        self.gradient[k] = self.Y[k] - y_hat
        # Append the index k to support 
        self.support = np.append(self.support, k)
        # If classified in positive class
        if self.Y[k] == 1:
        	# Do all this nonsense
            i = k
            # Godbless np.where 
            j = self.support[np.argmin(self.gradient[self.support[np.where(self.alpha[self.support] > self.A[self.support])]])]
        else:
        	# Otherwise, do this nonsense
            j = k
            i = self.support[np.argmax(self.gradient[self.support[np.where(self.alpha[self.support] < self.B[self.support])]])]
        if not self.__tauViolation(i, j):
            return -1
        self.__update(i, j)

    # LASVM Reprocess method
    # Most of this is copy/pasted from above lol 
    def __reprocess(self):
        i = self.support[np.argmax(self.gradient[self.support[np.where(self.alpha[self.support] < self.B[self.support])]])]
        j = self.support[np.argmin(self.gradient[self.support[np.where(self.alpha[self.support] > self.A[self.support])]])]
        if not self.__tauViolation(i, j):
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
        self.bias = float(self.gradient[i] + self.gradient[j])/2
        self.delta = self.gradient[i] - self.gradient[j]

    # Initialize seeds 
    def __seed(self):
        index = 1
        self.support = np.append(self.support, 0)
        # Set number of seeds - generally, more = better 
        seed_num = 40
        while list(self.Y[self.support]).count(1) < seed_num or list(self.Y[self.support]).count(-1) < seed_num:
            self.support = np.append(self.support, index)
            index += 1

    def __calcGrad(self):
        for i in range(self.numsamples):
            y_hat = 0
            for j in range(self.numsamples):
                y_hat += self.alpha[j] * stupid(self.X[j], self.X[i])
            self.gradient = np.append(self.gradient, Y[i] - y_hat)

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.alpha = np.zeros(Y.shape, dtype=np.float64)
        for y in self.Y:
            self.A = np.append(self.A, min(0, self.C * y))
            self.B = np.append(self.B, max(0, self.C * y))
        self.support = np.empty(0, dtype=int)
        self.gradient = np.empty(0)
        self.__seed()
        self.__calcGrad()
        for n in range(self.iterations):
            self.__process(random.randint(0, self.numsamples-1))
            self.__reprocess()
        while self.delta >= self.tau:
            if self.__reprocess() == -1:
                break

    def predict(self, X):
        predict = np.empty(0)
        for i,x in enumerate(X):
            res = 0
            for i in self.support:
                res += self.alpha[i] * stupid(x, self.X[i])
            if res == 0:
                predict = np.append(predict, -1)
            else:
                predict = np.append(predict, np.sign(res))
        return predict

###################################################################################

# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# randomizes the data
temp = shuffle(X,Y)
X = temp[0]
Y = temp[1]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 100
# numsamples = X.shape[0]
numsamples = 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
# k = KernelPerceptron(numsamples)
# k.fit(X,Y)
# k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

# bk = BudgetKernelPerceptron(beta, N, numsamples)
# bk.fit(X, Y)
# bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
# os.chdir('./sk_times/rbf')
# k = KernelPerceptron(numsamples)

###################################################################################
#																				  #
#                    SCRIPTS FOR FINDING RUNTIME AND ACCURACY     		          #
#																				  #
###################################################################################

# Runtime for KernelPerceptron
# iterations = 10
# fname = 'sk_'+str(numsamples)+'_' + str(iterations)+'_.csv'

# os.chdir('./sk_times')
# with open(fname, 'wb') as outcsv:
# 	writer = csv.writer(outcsv)
# 	writer.writerow(['iteration', 'seconds'])
# 	outcsv.close()

# with open(fname, 'a') as fl:
# 	writer = csv.writer(fl)
# 	for i in range(1,iterations + 1):
# 		temp = shuffle(X,Y)
# 		X = temp[0]
# 		Y = temp[1]
# 		print i 
# 		start = time.time()
# 		k.fit(X,Y)
# 		end = time.time() - start
#  		writer.writerow([i, end])
#  		print "Time: " + str(end)

# Accuracy for KernelPerceptron
# os.chdir('./accuracy')
# iterations = 5 
# fname = 'sk_'+str(numsamples)+'_' + 'stoopid_.csv'

# with open(fname, 'wb') as outcsv:
# 	writer = csv.writer(outcsv)
# 	writer.writerow(['iteration', 'accuracy', '# of sv'])
# 	outcsv.close()

# with open(fname, 'a') as fl:
# 	writer = csv.writer(fl)
# 	for i in range(1,iterations + 1):
# 		print i
# 		temp = shuffle(X,Y)
# 		X = temp[0]
# 		Y = temp[1]
# 		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)
		# I kinda started to lose it at this point lol
# 		print "STARTING THIS STEEEEEP"
# 		k.fit(X_train,y_train)
# 		print "DOOOOONE WITH THIS STEEEEEP"
# 		predict = k.predict(X_test)
# 		print "LET'S GET CHECKIIING"
# 		sv = k.sv()
# 		count = 0 
# 		for j in range(y_test.shape[0]):
# 			if j % 1000 == 0:
# 				print j
# 			if predict[j] == y_test[j]:
# 				count += 1
#  		writer.writerow([i, float(count)/y_test.shape[0], sv])
#  		print i, float(count)/y_test.shape[0], sv

###################################################################################

# Runtime for BudgetKernelPerceptron
# os.chdir('./bk_times')
# iterations = 10
# fname = 'bk_'+str(beta)+'_'+str(N)+'_'+str(numsamples)+'_' + str(iterations)+'_.csv'

# with open(fname, 'wb') as outcsv:
# 	writer = csv.writer(outcsv)
# 	writer.writerow(['iteration', 'seconds'])
# 	outcsv.close()

# with open(fname, 'a') as fl:
# 	writer = csv.writer(fl)
# 	for i in range(1,iterations + 1):
# 		temp = shuffle(X,Y)
# 		X = temp[0]
# 		Y = temp[1]
# 		print i 
# 		start = time.time()
# 		bk.fit(X,Y)
# 		end = time.time() - start
#  		writer.writerow([i, end])
#  		print "Time: " + str(end)

# Accuracy for BudgetKernelPerceptron 
# os.chdir('./accuracy')
# iterations = 5 
# fname = 'bk_'+str(beta)+'_'+str(N)+'_'+str(numsamples)+'_' + 'stoopid.csv'

# with open(fname, 'wb') as outcsv:
# 	writer = csv.writer(outcsv)
# 	writer.writerow(['iteration', 'accuracy'])
# 	outcsv.close()

# with open(fname, 'a') as fl:
# 	writer = csv.writer(fl)
# 	for i in range(1,iterations + 1):
# 		print i
# 		temp = shuffle(X,Y)
# 		X = temp[0]
# 		Y = temp[1]
# 		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)
# 		print "STARTING THIS STEEEEEP"
# 		bk.fit(X_train,y_train)
# 		print "DOOOOONE WITH THIS STEEEEEP"
# 		predict = bk.predict(X_test)
# 		print "LET'S GET CHECKIIING"
# 		count = 0 
# 		for j in range(y_test.shape[0]):
# 			if predict[j] == y_test[j]:
# 				count += 1
#  		writer.writerow([i, float(count)/y_test.shape[0]])
#  		print i, float(count)/y_test.shape[0]

###################################################################################

# Runtime for SMO and LASVM
# start = time.time()
# lasvm.fit(X,Y)
# end = time.time() - start
# print end 

# Accuracy for SMO and LASVM
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)
# print "STARTING THIS STEEEEEP"
# lasvm.fit(X_train,y_train)
# print "DOOOOONE WITH THIS STEEEEEP"
# predict = lasvm.predict(X_test)
# print "LET'S GET CHECKIIING"
# count = 0 
# for j in range(y_test.shape[0]):
#     if predict[j] == y_test[j]:
#         count += 1
# print float(count)/y_test.shape[0]
