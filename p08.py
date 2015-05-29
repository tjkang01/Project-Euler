# coding=utf-8 
import math 
import operator
import time

# The four adjacent digits in the 1000-digit number that have the greatest product are 9 × 9 × 8 × 9 = 5832.
# Find the thirteen adjacent digits in the 1000-digit number that have the greatest product. 
# What is the value of this product?

def fxn8():
	# gets file with very long number
	f = open(raw_input('Enter a file name: '), "r")
	number = map(int,f.read())
	f.close()
	length = int(raw_input('How many adjacent digits: '))
	max_val = 1

	start = time.clock()

	for i in range(len(number) - length + 1):
		temp = number[i:i+length]
		# potential optimization: if 0 detected, skip forward some indices forward
		if 0 not in temp:
			temp_val = reduce(operator.mul,temp,1)
			if temp_val > max_val:
				max_val = temp_val

	end = time.clock()			
	print(str(max_val) + " in " + str(end-start) + " seconds. ")