# coding=utf-8 
import math 
import operator
import time

# The four adjacent digits in the 1000-digit number that have the greatest product are 9 × 9 × 8 × 9 = 5832.
# Find the thirteen adjacent digits in the 1000-digit number that have the greatest product. 
# What is the value of this product?

def fxn8():
	# number is saved in txt file called "p08.txt"
	f = open(raw_input('Enter a file name: '), "r")
	number = map(int,f.read())
	f.close()
	length = int(raw_input('How many adjacent digits: '))
	max_val = 1

	start = time.clock()
	for i in range(len(number) - length + 1):
		temp = number[i:i+length]
		if 0 in temp: 
			temp = number[i + 12:i + 12 + length]
			if 0 not in temp:
				temp_val = reduce(operator.mul,temp,1)
				if temp_val > max_val:
					max_val = temp_val
	print(str(max_val) + " in " + str(time.clock()-start) + " seconds. ")