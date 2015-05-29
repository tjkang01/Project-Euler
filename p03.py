import math
import time

# The prime factors of 13195 are 5, 7, 13 and 29.
# What is the largest prime factor of the number 600851475143?

# 1. This is pretty bashy, as it just starts from 2 and goes up until the number can't be divided anymore 
# 2. More "sophisticated" methods of finding the actual prime values and employing those took too long
# 3. This returns both the prime factorization and the largest prime factor
def fxn3(x):  
 	start = time.clock()
	base = 2; product = 1; placeholder = x; primes = []; freq = []; count = 0; 
	while product != x:
		while(placeholder % base == 0):
			if base not in primes:
				primes.append(base)
			placeholder /= base
			product *= primes[-1]
			count += 1
		if count != 0:		
			freq.append(count)	
		base += 1
		count = 0
	factorization = dict(zip(primes, freq))
	end = time.clock()
	print(str(factorization) + " in " + str(end-start) + " seconds. ")	
