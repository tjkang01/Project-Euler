import math
import time

# The prime factors of 13195 are 5, 7, 13 and 29.
# What is the largest prime factor of the number 600851475143?

# returns prime factorization
def fxn3(x):  
 	start = time.clock()
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
	prime_factorization = dict((primes, freq))
	print(str(prime_factorization) + " in " + str(time.clock()-start) + " seconds. ")	