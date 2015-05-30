import math
import time

# Find the maximum total from top to bottom of the triangle in the text file p18.tzt

def fxn18():
	with open(raw_input('Enter a filename:'), "r") as f:
   		content = f.read().splitlines()
	numbers = [i.split() for i in content]
	numbers = [[int(j) for j in i] for i in numbers]

	start = time.clock()
	# goes bottom-up
	for i in range(len(numbers) - 2, -1,-1):
		# replaces the values of the triangle to be the partial maximum sums 
		# the value at the top is the maximum sum
		for j in range(i+1):
			numbers[i][j] += max(numbers[i+1][j],numbers[i+1][j+1])

	max_total = ' '.join(str(x) for x in max(numbers))
	print(str(max_total) + " in " + str(time.clock() - start) + " seconds. ")