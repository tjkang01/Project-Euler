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
   f.close() 
   length = int(raw_input('How many adjacent digits: '))
   numbers = [i.split() for i in content]
   numbers = [[int(j) for j in i] for i in numbers]
   temp = []; temp1 = []; temp2 = []; max_val = 1
   start = time.clock()

   # checks horizontal and vertical
   for j in range(len(numbers)):
      for k in range(len(content) - length + 1):
         max_val = max(max_val, reduce(operator.mul,numbers[j][k:k+length],1))
         for l in range(length):
            temp.append(numbers[k+l][j])
            if len(temp) == length:
               max_val = max(max_val, reduce(operator.mul, temp,1))
         temp = []      

   # checks diagonal going downwards
   for m in range(len(content) - length + 1):
      for n in range(len(content) - length + 1):
         for o in range(length):   
            temp1.append(numbers[m+o][n+o])
            if len(temp1) == length:
               max_val = max(max_val, reduce(operator.mul, temp1, 1))
         temp1 = []      
    
   # checks diagonal going upwards
   for q in range((length - 1), len(content)):
      for r in range (len(content) - length + 1):
         for s in range(length):
            temp2.append(numbers[q-s][r+s]) 
            if len(temp2) == length:
               max_val = max(max_val,reduce(operator.mul,temp2,1))
         temp2 = []                             

   print (str(max_val) + " in " + str(time.clock() - start) + " seconds. ")