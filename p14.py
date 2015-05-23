# coding=utf-8 
import math
import time

# The following iterative sequence is defined for the set of positive integers:
# n → n/2 (n is even)
# n → 3n + 1 (n is odd)

# Using the rule above and starting with 13, we generate the following sequence:
# 13 → 40 → 20 → 10 → 5 → 16 → 8 → 4 → 2 → 1
# It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. 
# Although it has not been proved yet (Collatz Problem), 
# it is thought that all starting numbers finish at 1.

# Which starting number, under one million, produces the longest chain?

def fxn14(x):
	number = x
	max_chain = []
	current_chain = []


	while number > 0:
		current_chain = [number]
		temp = number
		while temp != 1:
			if temp % 2 == 0:
				temp = temp/2
				current_chain.append(temp)
			else:
				temp = 3*temp +1
				current_chain.append(temp)
			# set up a count instead of length 	
		if len(current_chain) > len(max_chain):
			max_chain = current_chain
		number -= 1
		print max_chain, number	


