import math
import time

numbers = [1,2,3,4,5,6,7,8,9]

def isPandigital(x):
	digits = [int(digit) for digit in x]
	if len(set(digits)) == len(digits):
		if set(numbers) == set(digits):
			return True

def start_val(x):
	if x > 9:
		return 123
	else:
		return 1234

def fxn32():
	answers = []
	start = time.clock()
	for i in range(2,100):
		for j in range(start_val(i), 10000/i + 1):
			a = i; b = j; product = a * b
			temp = str(a) + str(b) + str(product)
			if isPandigital(temp):
				answers.append(product)
	print (str(sum(set(answers))) + " in " + str(time.clock() - start) + " seconds.")		