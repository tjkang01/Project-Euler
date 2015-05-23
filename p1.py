import math 
import time 

# If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. The sum of these multiples is 23.
# Find the sum of all the multiples of 3 or 5 below 1000.

# 1. calculate the number of multiples for each given integer 
# 2. use Gauss' method of addition to find the sum of the multiples 
# 3. subtract the overlap

# finds the sum of all multiples of any two given integers below a given value
def fxn1(x, multiple1, multiple2): 
	start = time.clock()
	# finds the LCM if the two numbers are even
	if (multiple1 % 2 == 0) and (multiple2 % 2 == 0):
		# checks to see if one number is just a multiple of the other
		if (multiple2 % multiple1 == 0):
			multiple3 = multiple2 
		elif (multiple1 % multiple2 ==0):
			multiple3 = multiple1
		# LCM between two even numbers that are not multiples is the product/2		
		else:	
			multiple3 = (multiple1 * multiple2)/2			
	else:
		multiple3 = multiple1 * multiple2	
	# calculates the number of times each multiple has to be repeated - i.e. 1000 would result in floor1 = 333 and floor2 = 199 
	# as such, the list of multiples would be [3*1, 3*2 ..., 3*333] and [5*1, 5*2, ..., 5 *199]
	floor1 = int(math.floor((x-1)/multiple1))
	floor2 = int(math.floor((x-1)/multiple2))
	# uses Gauss' method of addition and multiplies by provided multiple 
	sum1 = ((floor1*(floor1+1))/2)*(multiple1)
	sum2 = ((floor2*(floor2+1))/2)*(multiple2)
	# takes into acoount if inputted value is greater than LCM
	if x >= (multiple3): 
		# subtracts the overlap 
		floor3 = int(math.floor((x-1)/multiple3))
		sum3 = ((floor3*(floor3+1))/2)*(multiple3)
		end = time.clock()
		print(str((sum1+sum2)-sum3) + " in " + str(end-start) + " seconds. ")
	else:
		end = time.clock()
		print(str(sum1+sum2) + " in " + str(end-start) + " seconds. ")