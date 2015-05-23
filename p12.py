import math
import time

# The sequence of triangle numbers is generated by adding the natural numbers. 
# So the 7th triangle number would be 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28. The first ten terms would be:
# 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, ...
# Let us list the factors of the first seven triangle numbers:

#  1: 1
#  3: 1,3
#  6: 1,2,3,6
# 10: 1,2,5,10
# 15: 1,3,5,15
# 21: 1,3,7,21
# 28: 1,2,4,7,14,28
# 36: 1,2,3,4,6,6,9,12,18,36
# 45: 1,3,5,9,15,45

# We can see that 28 is the first triangle number to have over five divisors.

# What is the value of the first triangle number to have over five hundred divisors?

def fxn12(x):
	index = 1
	count = 2
	start = time.clock()

	while count <= x:
		index += 1
		count = 2
		triangular_number = (index * (index + 1))/2
		for i in range(2, int(math.ceil(triangular_number**0.5)) + 1):
			if triangular_number % i == 0:
				count += 2
				
	print(str(triangular_number) + " in " + str(time.clock() - start) + " seconds. ")
