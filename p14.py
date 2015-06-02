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

def collatz(n):
	if n % 2 == 0:
		return n/2
	else:
		return 3*n + 1		

def length(y, record = {1:1}):
	if y not in record: 
		record[y] = length(collatz(y)) + 1
	return record[y]	

def fxn14():
	x = int(raw_input("Enter limit: "))
	
	start = time.clock()
	print(str(max(range(1,x), key=length)) + " in " + str(time.clock() - start) + " seconds. ")