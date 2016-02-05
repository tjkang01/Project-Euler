import math
import time

# The prime factors of 13195 are 5, 7, 13 and 29.
# What is the largest prime factor of the number 600851475143?

# returns prime factorization
def fxn3():  
	x = int(raw_input("Enter number: "))
	
 	start = time.clock()
 	base = 2; product = 1; placeholder = x; factorization = {};
 	while product != x:
 		count = 0
 		if (placeholder % base == 0):
 			while (placeholder % base == 0):
 				placeholder /= base
 				product *= base
 				count += 1
 			factorization[base] = count
 		base += 1
	print(str(factorization) + " in " + str(time.clock()-start) + " seconds. ")	