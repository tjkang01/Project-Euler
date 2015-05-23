# coding=utf-8 
import math
import time
import operator

# In the 20×20 grid below, four numbers along a diagonal line have been marked in red.
# The product of these numbers is 26 × 63 × 78 × 14 = 1788696.
# What is the greatest product of four adjacent numbers in the same direction 
# (up, down, left, right, or diagonally) in the 20×20 grid?

def fxn11():
	with open(raw_input('Enter a filename:'), "r") as f:
   		content = f.read().splitlines()
   	length = int(raw_input('How many adjacent digits: '))
   	temp1 = []
   	temp2 = []
   	temp3 = []
   	max_val = 1
   	numbers = [i.split() for i in content] 
   	numbers = [[int(j) for j in i] for i in numbers]

   	start = time.clock()
   	for j in range(len(numbers)):
   		for k in range(len(content) - length + 1):
   			temp = numbers[j][k:k+length]
  	 		if 0 not in temp:
  	 			temp_val = reduce(operator.mul,temp,1)
   	 			if temp_val > max_val:
				 	max_val = temp_val
   			for l in range(length):
   				temp1.append(numbers[k+l][j])
   				if len(temp1) == length:
   					if 0 not in temp1:
   						temp_val = reduce(operator.mul,temp1,1)
   						if temp_val > max_val:
								max_val = temp_val
   			temp1 = []

   	for m in range(len(content) - length + 1):
   	 	for n in range (len(content) - length + 1):
   	 		for o in range(length):
   	 			temp2.append(numbers[m+o][n+o])
   	 			if len(temp2) == length:
   	 				if 0 not in temp2:
   	 					temp_val = reduce(operator.mul,temp2,1)
   	 					if temp_val > max_val:
   	 						max_val = temp_val
   	 		temp2 = []
   	 		
   	for q in range((length - 1), len(content)):
   		for r in range (len(content) - length + 1):
   			for s in range(length):
   				temp3.append(numbers[q-s][r+s]) 
   				if len(temp3) == length:
   					if 0 not in temp3:
   						temp_val = reduce(operator.mul,temp3,1)
   						if temp_val>max_val:
   							max_val = temp_val
   							print temp3
   			temp3 = []										
   					
   	print (str(max_val) + " in " + str(time.clock() - start) + " seconds. ")