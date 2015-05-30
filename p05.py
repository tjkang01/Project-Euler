import math
import time
# 2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.
# What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?

# calculates prime factorization of the desired number
def fxn5():
	start = time.clock()
	if lbound == 1:
		lbound += 1
	divisors = range(lbound, ubound + 1)
	base = 2; product = 1; primes = []; freq = []; count = 0; factorization = {}; total = 1
	for i in divisors:
		placeholder = i
		while product != i:
			while (placeholder % base == 0):
				if base not in primes:
					primes.append(base)
				placeholder /= base
				product *= primes[-1]
				count += 1
			if count != 0:
				freq.append(count)
			base += 1
			count = 0
		product = 1; base = 2; count = 0
		temp = dict(zip(primes,freq))
		primes = []
		freq = []
		for key, val in temp.items():
			if key not in factorization.keys():
				factorization[key] = val
			else:
				if factorization.get(key) < temp.get(key): 
					factorization[key] = factorization.get(key) + 1
	for keys, vals in factorization.items():
		total *= (keys ** vals)
	print(str(factorization) + " or " + str(total) + " in " + str(time.clock()-start) + " seconds. ")