# coding=utf-8
import math
import time

# Starting in the top left corner of a 2×2 grid, and only being able to move to the right and down, there are exactly 6 routes to the bottom right corner.
# How many such routes are there through a 20×20 grid?

# Because the problem limits movement to some combination of right and down, it boils down to simple combinatorics.
# For any given grid, it takes 2 * (length of one side) steps to go from one corner to another

def fxn15():
	x = int(raw_input("Enter length of one side of square grid: "))
	
	start = time.clock()
	total = math.factorial(x * 2)
	overlap = math.factorial(x) * math.factorial(x)
	print(str(total/overlap) + " in " + str(time.clock()-start) + " seconds. ")