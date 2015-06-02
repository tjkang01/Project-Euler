import math
import time

# Generates a list of primes in primes.txt 
# Left running while solving other problems

def primality(y):
	for i in xrange(2,int(y**0.5) + 1):
		if y % i == 0:
			return False
	return True
	
def primes():
	number = 2
	index = 1
	f = open('primes.txt', 'w')
	while True:
		if primality(number):
			f.write("%s %s\n" % (index, number))
			index += 1
		number += 1
	f.close()