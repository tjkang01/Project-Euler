import math
import time

# Find the value of d < 1000 for which 1/d contains the longest recurring cycle in its decimal fraction part.

def has_cycle(x):
	if x % 2 == 0 or x % 5 == 0:
		return False
	else:
		return True

def cycle_length(x):
	if has_cycle(x):
		count = 1
		rem = 9
		while rem > 0:
			rem = (10 * rem + 9) % x
			count += 1
		return count
	else:
		return 0				

def fxn26():
	upper = int(raw_input("Enter upper limit: "))
	start = time.clock()
	num = 0
	max_len = 0
	for i in range(2,upper):
		temp_length = cycle_length(i)
		if temp_length > max_len:
			num = i
			max_len = temp_length
	print(str(num) + " has cycle of length " + str(max_len) + " in " + str(time.clock() - start) + " seconds. ")