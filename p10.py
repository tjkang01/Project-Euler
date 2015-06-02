import math
import time

# The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
# Find the sum of all the primes below two million.

def primality(y):
	for i in xrange(2,int(y**0.5) + 1):
		if y % i == 0:
			return False
	return True	

def fxn10():
	x = int(raw_input("Enter the limit: "))

	start = time.clock()
	total = 0
	for i in range(2,x):
		if primality(i):
			total += i
	print (str(total) + " in " + str(time.clock()-start) + " seconds. ")