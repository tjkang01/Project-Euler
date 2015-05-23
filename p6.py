import math
import time 

# lolz
def fxn6(x):
	start = time.clock()
	sum1 = (x*(x+1)*(2*x+1))/6
	sum2 = (x*(x+1))/2
	total = (sum2 ** 2) - (sum1)
	end = time.clock()
	print(str(total) + " in " + str(end-start) + " seconds. ")