import math
import time

# 2^15 = 32768 and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26.
# What is the sum of the digits of the number 2^1000?

def fxn16():
	x = int(raw_input("Enter base: "))
	y = int(raw_input("Enter power: "))

	start = time.clock()
	number = str(x**y)
	total = 0 
	for val in number:
		total += int(val)

	print(str(total) + " in " + str(time.clock() - start) + " seconds. ")