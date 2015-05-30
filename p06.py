import math
import time 

# Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.

def fxn6(x):
	start = time.clock()
	sum1 = (x*(x+1)*(2*x+1))/6
	sum2 = (x*(x+1))/2
	total = (sum2 ** 2) - (sum1)
	print(str(total) + " in " + str(time.clock()-start) + " seconds. ")