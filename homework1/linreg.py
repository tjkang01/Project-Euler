#####################
# CS 181, Spring 2016
# Homework 1, Problem 3
#
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'congress-ages.csv'
times  = []
ages = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        times.append(float(row[0]))
        ages.append(float(row[1]))

# Turn the data into numpy arrays.
times  = np.array(times)
ages = np.array(ages)

# Plot the data.
plt.plot(times, ages, 'o')
plt.xlabel("Congress age (nth Congress)")
plt.ylabel("Average age")
plt.show()

# Create the simplest basis, with just the time and an offset.
# This needs to be changed
# X = np.vstack((np.ones(times.shape), times)).T

def basis_fxn_a(x):
	ones = [np.ones(x.shape)]
	for i in xrange(1,7):
		ones.append(x ** i)
	basis = np.vstack(ones).T
	return basis

def basis_fxn_b(x):
	ones = [np.ones(x.shape)]
	for i in xrange(1,3):
		ones.append(x ** i)
	basis = np.vstack(ones).T
	return basis

def basis_fxn_c(x):
	ones = [np.ones(x.shape)]
	for i in xrange(1,4):
		ones.append(np.sin(x/i))
	basis = np.vstack(ones).T
	return basis

def basis_fxn_d(x):
	ones = [np.ones(x.shape)]
	for i in xrange(1,7):
		ones.append(np.sin(x/i))
	basis = np.vstack(ones).T
	return basis

def basis_fxn_e(x):
	ones = [np.ones(x.shape)]
	for i in xrange(1,20):
		ones.append(np.sin(x/i))
	basis = np.vstack(ones).T
	return basis

X = basis_fxn_e(times)
# Nothing fancy for outputs.
Y = ages

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_times!!!!!
grid_times = np.linspace(75, 120, 200)
# grid_X = np.vstack((np.ones(grid_times.shape), grid_times))
grid_X = basis_fxn_e(grid_times)
# originally grid_X.T
grid_Yhat  = np.dot(grid_X, w)

# Plot the data and the regression line.
plt.plot(times, ages, 'o', grid_times, grid_Yhat, '-')
plt.xlabel("Congress age (nth Congress)")
plt.ylabel("Average age")
plt.show()

