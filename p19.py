import math
import time

# You are given the following information, but you may prefer to do some research for yourself.
# 1 Jan 1900 was a Monday.
# Thirty days has September,
# April, June and November.
# All the rest have thirty-one,
# Saving February alone,
# Which has twenty-eight, rain or shine.
# And on leap years, twenty-nine.
# A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.
# How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?


def fxn19(start, end, desired_day):
	start_time = time.clock()
	count = 0
	# 1 Jan 1901 was a Tuesday
	day = 2
	for year in range(start, end + 1):
		if (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0)):
			months = [31,29,31,30,31,30,31,31,30,31,30,31]
		else:
			months = [31,28,31,30,31,30,31,31,30,31,30,31]
		
		for month in range(12):
			if day % 7 == desired_day:
				count += 1
			day = day + months[month] % 7
			
	print (str(count) + " in " + str(time.clock() - start) + " seconds. ")				

	