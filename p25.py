import math
import time

# What is the index of the first term in the Fibonacci sequence to contain 1000 digits?

# Large fibs can be calculated as (phi^n)/sqrt(5)

def fxn25():
	digits = int(raw_input("Enter number of digits: "))
	limit = digits-1
	phi = 1.6180
	index = math.floor((math.log(10) * 999 + math.log(5)/2)/math.log(phi))
	print index