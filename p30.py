import math
import time

def fxn30():
	power = int(raw_input("Enter the power: "))
	start = time.clock()
	terms = []
	for i in range(2,200000):
		temp = [int(x)**power for x in str(i)]
		if sum(temp) == i:
			terms.append(i)
	print terms		
	print (str(sum(terms)) + " in " + str(time.clock()- start) + " seconds.")			 	
		