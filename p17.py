import math
import time
import inflect

# If the numbers 1 to 5 are written out in words: one, two, three, four, five, then there are 3 + 3 + 5 + 4 + 4 = 19 letters used in total.
# If all the numbers from 1 to 1000 (one thousand) inclusive were written out in words, how many letters would be used?

# okay, I cheated a bit with the inflect library, but no shame.
def fxn17():
	x = int(raw_input("Enter lower bound: "))
	y = int(raw_input("Enter upper bound: "))
	
	start = time.clock()
	p = inflect.engine()
	words = []
	total = 0 

	for i in range(x,y+1):
		words.append(p.number_to_words(i))

	for word in words:
		for letter in word:
			if letter != ' ' and letter != '-' and letter != ',':
				total += 1
	print(str(total) + " in " + str(time.clock() - start ) + " seconds. ")		