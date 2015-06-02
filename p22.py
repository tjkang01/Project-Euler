# coding=utf-8
import math
import time
import re

# Using names.txt (right click and 'Save Link/Target As...'), a 46K text file containing over five-thousand first names, begin by sorting it into alphabetical order. Then working out the alphabetical value for each name, multiply this value by its alphabetical position in the list to obtain a name score.
# For example, when the list is sorted into alphabetical order, COLIN, which is worth 3 + 15 + 12 + 9 + 14 = 53, is the 938th name in the list. So, COLIN would obtain a score of 938 × 53 = 49714.
# What is the total of all the name scores in the file?

def fxn22():
	f = open(raw_input('Enter a file name: '), "r")
	names = sorted([name.replace('"','') for name in f.read().split(',')])
	f.close()

	start = time.clock()
	index = 1
	total_of_list = 0

	for name in names:
		total_of_name = 0
		for letter in name:
			letter = str.lower(letter)
			value = ord(letter) - 96
			total_of_name += value
		total_of_list += index * total_of_name
		index += 1
	print (str(total_of_list) + " in " + str(time.clock() - start) + " seconds. ")