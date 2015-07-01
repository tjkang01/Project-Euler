import math
import time

# A permutation is an ordered arrangement of objects. For example, 3124 is one possible permutation of the digits 1, 2, 3 and 4. If all of the permutations are listed numerically or alphabetically, we call it lexicographic order. The lexicographic permutations of 0, 1 and 2 are:
# 012   021   102   120   201   210
# What is the millionth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?

def fxn24():
	first = int(raw_input("Enter first number: "))
	last = int(raw_input("Enter last number: "))
	perm = int(raw_input("Enter desired index: ")) - 1

	start = time.clock()
	num_to_add = range(first,last)
	permut = range(first,last)
	number = ''

	while len(permut) != 0:
		temp = permut[-1]
		permut.remove(temp)
		index = perm/math.factorial(temp)
		number = number + str(num_to_add[index])
		num_to_add.remove(num_to_add[index])
		perm = perm % math.factorial(temp)
	print(number + " in " + str(time.clock() - start) + " seconds. ")	