# coding=utf-8
import math
import time

# A palindromic number reads the same both ways
# The largest palindrome made from the product of two 2 digit numbers is 9009 = 91 × 99
# Find the largest palindrome made from the product of two 3 digit numbers

# 1. Multiplies all the numbers within the given bounds by each other
# 2. Checks if the number is palindromic through slicing
# 3. Stores, records, and orders the palindromes in increasing order 
def fxn4(lbound,ubound):
	start = time.clock()
	palindromes = [] 
	nums = range(lbound, ubound+1)
	first_multiple = nums
	second_multiple = nums 
	for i in first_multiple:
		for n in second_multiple:
			product = i * n
			# uses slicing to check if the number is equal to itself reversed 
			if str(product) == str(product)[::-1]:
				if product not in palindromes:
					palindromes.append(product)
	palindromes = sorted(palindromes, key = int)
	end = time.clock()
	print(str(palindromes) + " in " + str(end-start) + " seconds. ")
