import math 
import time 

# If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. The sum of these multiples is 23.
# Find the sum of all the multiples of 3 or 5 below 1000.
		
def LCM(x,y):
	if (x % 2 == 0) and (y % 2 == 0):
		if (y % x == 0):
			return y
		elif (x % y == 0):
			return x
		else:
			return (x * y)/2
	else:
		return x * y	

def Gauss(x):
	return (x * (x+1))/2

def fxn1(): 
	limit = int(raw_input("Enter the limit: "))
	multiple1 = int(raw_input("Enter first multiple: "))
	multiple2 = int(raw_input("Enter second multiple: "))

	start = time.clock()
	multiple3 = LCM(multiple1, multiple2)
	# calculates the number of times each multiple has to be repeated 
	floor1 = int(math.floor((limit-1)/multiple1))
	floor2 = int(math.floor((limit-1)/multiple2))
	# uses Gauss' method of addition and multiplies by provided multiple 
	sum1 = Gauss(floor1)*(multiple1)
	sum2 = Gauss(floor2)*(multiple2)
	# takes into account if inputted value is greater than LCM
	if limit >= (multiple3): 
		# subtracts the overlap 
		floor3 = int(math.floor((limit-1)/multiple3))
		sum3 = ((floor3*(floor3+1))/2)*(multiple3)
		print(str((sum1+sum2)-sum3) + " in " + str(time.clock()-start) + " seconds. ")
	else:
		print(str(sum1+sum2) + " in " + str(time.clock()-start) + " seconds. ")