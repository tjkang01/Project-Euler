import math
import time 

# Each new term in the Fibonacci sequence is generated by adding the previous two terms. 
# By starting with 1 and 2, the first 10 terms will be:
# 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
# By considering the terms in the Fibonacci sequence whose values do not exceed four million, find the sum of the even-valued terms.

# even-valued terms
# limit is the value which the last term of the Fibonacci sequence cannot exceed
def fxn2e():
	limit = int(raw_input("Enter limit: "))
	start1 = int(raw_input("Enter first Fibonacci number: "))
	start2 = int(raw_input("Enter second Fibonacci number: "))

	start = time.clock()
	s = 0
	computed = {0: start1, 1: start2}
	# checks to see if the starting numbers needed to be added to the count
	# in reality, the pair of starting numbers should only be (0,1), (1,1) or (1,2)
	if (start1 % 2 == 0) and (start2 % 2 == 0):
		evens = [start1, start2]
	elif (start1 % 2 == 0 ):
		evens = [start1] 
	elif (start2 % 2 ==0):
		evens = [start2]
	else:
		evens = []		
	for n in range(2, limit+1):
		# employs memoization to increase efficiency
		computed[n] = computed[n-1] + computed[n-2]
		last_dig = computed.values()[-1]
		if last_dig < limit: 
			if (last_dig % 2) == 0:
				evens.append(last_dig)
		else: 
			break
	print(str(sum(evens)) + " in " + str(time.clock()-start) + " seconds. ")	

# odd-valued terms
def fxn2o():
	limit = int(raw_input("Enter limit: "))
	start1 = int(raw_input("Enter first Fibonacci number: "))
	start2 = int(raw_input("Enter second Fibonacci number: "))
	
	start = time.clock()
	s = 0
	computed = {0: start1, 1: start2}
	# checks to see if the starting numbers needed to be added to the count
	# in reality, the pair of starting numbers should only be (0,1), (1,1) or (1,2)
	if (start1 % 2 == 1) and (start2 % 2 == 1):
		odds = [start1, start2]
	elif (start1 % 2 == 1):
		odds = [start1] 
	elif (start2 % 2 ==1):
		odds = [start2]
	else:
		odds = []		
	for n in range(2, limit+1):
		# employs memoization to increase efficiency
		computed[n] = computed[n-1] + computed[n-2]
		last_dig = computed.values()[-1]
		if last_dig < limit: 
			if (last_dig % 2) == 1:
				odds.append(last_dig)
		else: 
			break		
	print(str(sum(odds)) + " in " + str(time.clock()-start) + " seconds. ")	

# prints out the Fibonacci sequence under a given value with starting terms (0,1), (1,1), or (1,2)
def fib(start1, start2, limit):
	computed = {0: start1, 1: start2}
	for n in range(2, limit+1):
		computed[n] = computed[n-1] + computed[n-2]
		last_dig = computed.values()[-1]
		if last_dig > limit: 
			del computed[n]
			break
	print computed.values()						