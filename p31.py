import math
import time

coins = [1,2,5,10,20,50,100,200]

def fxn31():
		amount = int(raw_input("Enter amount: "))
		start = time.clock()
		ways = [1] + [0]*amount
		for coin in coins:
			for i in range(coin, amount+1):
				ways[i] += ways[i-coin]
		print(str(ways[amount]) + " in" + str(time.clock() - start) + " seconds.")		
