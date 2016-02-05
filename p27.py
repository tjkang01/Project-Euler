import math
import time

def primality(y):
	for i in range(2,int(y**0.5) + 1):
		if y % i == 0:
			return 0
	return 1	

def fxn27():
	upper = int(raw_input("Enter upper limit: "))
	maxa = 0
	maxb = 0
	maxn = 0
	start = time.clock()
	for a in range(-upper + 1, upper):
		for b in range(-upper + 1, upper):
			n = 0
			while n*n + a*n + b > 1 and primality(n*n + a*n + b):
				n += 1
			if n > maxn:
				maxa, maxb = a,b
				maxn = n
				print maxa,maxb,maxn
	print (str(maxa*maxb) + " in " + str(time.clock()- start) + " seconds.")			