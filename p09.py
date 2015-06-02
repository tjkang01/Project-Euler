import math
import time

# A Pythagorean triplet is a set of three natural numbers, a < b < c, for which, a^2 + b^2 = c^2
# There exists exactly one Pythagorean triplet for which a + b + c = 1000.
# Find the product abc.

def fxn9():
	target_sum = int(raw_input("Enter the sum of the Pythagorean triplet: "))

	start = time.clock()
	upper = int(math.ceil((target_sum+1)/2 ** 0.5))
	for i in range(1, upper):
		for j in range(1, upper):
			k = target_sum - i - j
			if i ** 2 + j ** 2 == k ** 2:
				return (str(i*j*k) + " in " + str(time.clock()-start) + " seconds.")