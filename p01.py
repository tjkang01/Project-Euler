import math 
import time 

# If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. The sum of these multiples is 23.
# Find the sum of all the multiples of 3 or 5 below 1000.
		
def LCM(a,b):
	if (a % 2 == 0) and (b % 2 == 0):
		if (b % a == 0):
			return b
		elif (a % b == 0):
			return a
		else:
			return (a * b)/2
	else:
		return a * b	

# finds the sum of all multiples of any two given integers below a given value
def fxn1(x, multiple1, multiple2): 
	start = time.clock()
	multiple3 = Util.LCM(multiple1, multiple2)
	# calculates the number of times each multiple has to be repeated 
	floor1 = int(math.floor((x-1)/multiple1))
	floor2 = int(math.floor((x-1)/multiple2))
	# uses Gauss' method of addition and multiplies by provided multiple 
	sum1 = ((floor1*(floor1+1))/2)*(multiple1)
	sum2 = ((floor2*(floor2+1))/2)*(multiple2)
	# takes into account if inputted value is greater than LCM
	if x >= (multiple3): 
		# subtracts the overlap 
		floor3 = int(math.floor((x-1)/multiple3))
		sum3 = ((floor3*(floor3+1))/2)*(multiple3)
		end = time.clock()
		print(str((sum1+sum2)-sum3) + " in " + str(end-start) + " seconds. ")
	else:
		end = time.clock()
		print(str(sum1+sum2) + " in " + str(end-start) + " seconds. ")