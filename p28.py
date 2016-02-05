import math
import time

def fxn28():
	sides = int(raw_input("Enter number of sides: "))
	start = time.clock()
	iterations = (sides - 1)/2

	total = 1 
	for i in range(1, iterations + 1):
		total += 4*(2*i+1)**2 - 12*i

	print (str(total) + " in " + str(time.clock()- start) + " seconds.")			 	