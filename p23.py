import math
import time
import p21

# A perfect number is a number for which the sum of its proper divisors is exactly equal to the number. For example, the sum of the proper divisors of 28 would be 1 + 2 + 4 + 7 + 14 = 28, which means that 28 is a perfect number.
# A number n is called deficient if the sum of its proper divisors is less than n and it is called abundant if this sum exceeds n.
# As 12 is the smallest abundant number, 1 + 2 + 3 + 4 + 6 = 16, the smallest number that can be written as the sum of two abundant numbers is 24. By mathematical analysis, it can be shown that all integers greater than 28123 can be written as the sum of two abundant numbers. However, this upper limit cannot be reduced any further by analysis even though it is known that the greatest number that cannot be expressed as the sum of two abundant numbers is less than this limit.
# Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers.

def fxn23():
	limit = int(raw_input("Enter upper limit: "))
	start = time.clock()
	abundant = []
	total = 0
	# a value of True indicates it is the sum of two abundant numbers
	abundant_sum = {}

	for i in range(1,limit):
		abundant_sum[i] = False

	for i in range(1, limit+1):
			temp = p21.product_fxn(i)
			if temp > i:
				abundant.append(i)
	bound = len(abundant)		

	for i in range(0, bound):
		for j in range(i,bound):
				summed = abundant[i] + abundant[j] 
				if summed < limit:
					abundant_sum[summed] = True
				else:
					break

	for i in range(1,limit):
		if abundant_sum[i] == False:
			total += i

	print(str(total) + " in " + str(time.clock() - start) + " seconds. ")	