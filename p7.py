import math
import time

# By listing the first six prime numbers: 2,3,5,7,11, and 13, we can see that the 6th prime is 13. 
# What is the 10,001st prime number?

def primality(y):
	for i in xrange(2,int(y**0.5) + 1):
		if y % i == 0:
			return False
	return True	

def fxn7(x):
	start = time.clock()
	primes = []
	n = 1
	
	while len(primes) <= x:
		if primality(n):
			primes.append(n)
		n += 1	
	end = time.clock()	
	print (str(primes) + " in " + str(end-start) + " seconds. ")