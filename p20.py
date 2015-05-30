# coding=utf-8
import math
import time

# n! means n × (n − 1) × ... × 3 × 2 × 1

# For example, 10! = 10 × 9 × ... × 3 × 2 × 1 = 3628800,
# and the sum of the digits in the number 10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.

# Find the sum of the digits in the number 100!

def fxn20(x):
	start = time.clock()
	number = str(math.factorial(x))
	total = 0

	for digit in number:
		total += int(digit)

	print(str(total) + " in " + str(time.clock() - start) + " seconds. ")