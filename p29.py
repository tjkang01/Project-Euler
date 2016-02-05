import math
import time

def fxn29():
	lower = int(raw_input("Enter lower bound: "))
	upper = int(raw_input("Enter upper bound: "))
	start = time.clock()
	terms = [a**b for a in range(lower,upper+1) for b in range (lower,upper+1)]
	print (str(len(set(terms))) + " in " + str(time.clock()- start) + " seconds.")			 	
