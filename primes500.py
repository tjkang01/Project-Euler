import math
import time
import pickle 

# Generates a list of primes in primes.txt 
# Left running while solving other problems

def primality(y):
	for i in xrange(2,int(y**0.5) + 1):
		if y % i == 0:
			return False
	return True
	
def primes():
	limit = int(raw_input("Enter index of last prime: "))
	number = 2
	index = 1
	prime = {}
	while index < limit + 1:
		if primality(number):
			prime[index] = number
			index += 1
		number += 1
	pickle.dump(prime, open("primes500.p", "wb"))
	# pickle.load(open( "save.p", "rb" ))