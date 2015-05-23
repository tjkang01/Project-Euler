import math
import time

# Work out the first ten digits of the sum of the following one-hundred 50-digit numbers.

def fxn13():
	with open(raw_input('Enter a filename:'), "r") as f:
   		content = f.read().splitlines()
   	length = int(raw_input('How many digits: '))
   	start = time.clock()
   	numbers = [int(i) for i in content]
   	total = str(sum(numbers))
   	print (total[0:length] + " in " + str(time.clock() - start) + " seconds. ")