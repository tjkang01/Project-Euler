# coding=utf-8
import math
import time

# A palindromic number reads the same both ways
# The largest palindrome made from the product of two 2 digit numbers is 9009 = 91 × 99
# Find the largest palindrome made from the product of two 3 digit numbers

def is_palindrome(x):
	if str(x) == str(x)[::-1]:
		return True
	else:
		return False	

# returns all palindromes between the limits
def fxn4(lbound,ubound):
	start = time.clock()
	palindromes = [] 
	nums = range(lbound, ubound + 1)
	first_multiple = nums
	second_multiple = nums 
	for i in first_multiple:
		for n in second_multiple:
			product = i * n
			# uses slicing to check if the number is equal to itself reversed 
			if is_palindrome(product):
				if product not in palindromes:
					palindromes.append(product)
	palindromes = sorted(palindromes, key = int)
	print(str(palindromes) + " in " + str(time.clock()-start) + " seconds. ")
