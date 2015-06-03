# coding=utf-8
import math
import time

# Let d(n) be defined as the sum of proper divisors of n (numbers less than n which divide evenly into n).
# If d(a) = b and d(b) = a, where a ≠ b, then a and b are an amicable pair and each of a and b are called amicable numbers.
# For example, the proper divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110; therefore d(220) = 284. The proper divisors of 284 are 1, 2, 4, 71 and 142; so d(284) = 220.
# Evaluate the sum of all the amicable numbers under 10000.

def pfactorization(x):
	base = 2; product = 1; placeholder = x; primes = []; freq = []; count = 0; 
	while product != x:
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
	prime_factorization = dict(zip(primes, freq))
	return prime_factorization

# iterate over a prime factorization
def sum_of_divisor(x,y):
	return (x ** (y+1) - 1)/(x-1)

# sum of proper divisors
def product_fxn(x): 
	total = 1
	primes = pfactorization(x)
	for bases in primes.keys():
		total *= sum_of_divisor(bases,primes[bases])
	return total - x
	
def fxn21():
	x = int(raw_input("Enter limit: "))	
	start = time.clock()	
	numsum = 0

	for i in range(2,x + 1):
		temp = product_fxn(i)
		if temp != i:
			if i == product_fxn(temp):
				numsum += i
	print(str(numsum) + " in " + str(time.clock() - start) + " seconds. ")