# CS 181, Spring 2016
# Homework 4: Clustering
# Name: Timothy Kang
# Email: tkang01@college.harvard.edu

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import sys
import time

class KMeans(object):
    # K is the K in Kself.Means
    # useKMeansPP is a boolean. If True, you should initialize using Kself.Means++
    def __init__(self, K, useKMeansPP):
        self.K = K
        self.useKMeansPP = useKMeansPP

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        self.X = X
        number = X.shape[0]
        # Used in KMeans++
        self.dist = np.empty(number)
        # Responsibilities (which cluster each point is assigned to) - a bunch of hot-encoded vectors
        self.rs = np.zeros((number,K))
        # Probabilities for KMeansPP
        self.prob = np.empty(number)
        # Array that keeps track of means
        self.means = np.zeros((K,28,28))
        # Array that keeps track of value returned by objective function to be used in plotting
        self.obj_output = np.empty(0)

        # If we aren't using KMeansPP, initialize with random responsibilities
        if not self.useKMeansPP:
            for i in range(number):
                rand = random.randint(0, K - 1)
                self.rs[i][rand] = 1
        # Use KMeansPP algorithm
        else:
            # Initialize centers with 0 
            print "STARTING KMEANSPP"
            centers = [0]
            for k in range(1,K):
                for n in range(number):
                    closest_distancence = sys.maxint
                    for k_prime in range(k):
                        # Calculate Euclidian distancence
                        if np.linalg.norm(X[n] - self.means[k_prime]) < closest_distancence:
                            # Update closest distancence
                            closest_distancence = np.linalg.norm(X[n] - self.means[k_prime])
                    # Store it for calculation        
                    self.dist[n] = closest_distancence
                # Squared sum of distancences
                ss = np.sum(self.dist * self.dist)
                # Probability proportional to squared sum
                self.prob = (float(1) / ss) * (self.dist * self.dist)

                # Start with random value
                r = random.random()
                prob_sum = 0
                next_center = 0
                # Pick probabilities until conditions are satisfied to find next center
                while prob_sum < r and next_center < number:
                    prob_sum += self.prob[next_center]
                    next_center += 1

                # Update means 
                self.means[k] = X[next_center]  
                # Add new center to list
                centers.append(next_center)

            print "ALL DONE HERE"
            # Define responsbilities in terms of these centers 
            for index, value in enumerate(centers):
                self.rs[value][index] = 1 

            

        # Change keeps track of whether or not the responsibilties have changed
        change = True
        iteration = 0
        # Lloyd's Algorithm
        while change:
            change = False

            # Necessary for plotting objective function 
            objective_function_val = 0 
            for n in range(number):
                for k in range(K):
                    objective_function_val += self.rs[n][k] * np.linalg.norm(X[n] - self.means[k])
            self.obj_output = np.append(self.obj_output, objective_function_val)

            for k in range(K):
                ones = self.rs[:, [k]]
                # Checks how many responsbilities are assigned to this cluster
                size_of_cluster = np.sum(ones)
                value_of_cluster = np.zeros((28,28))
                for n in range(number):
                    if ones[n][0]:
                        value_of_cluster += X[n]
                # Update means vector
                self.means[k] = (float(1)/size_of_cluster) * value_of_cluster 
            for n in range(number):
                new_rs = np.zeros(K)
                k_new = -1 
                distancence = sys.maxint
                for k in range(K):
                    if np.linalg.norm(X[n] - self.means[k]) < distancence: 
                        distancence = np.linalg.norm(X[n] - self.means[k])
                        k_new = k
                # Update responsbilities
                new_rs[k_new] = 1
                # Check for changes
                if not np.array_equal(new_rs, self.rs[n]):
                    change = True
                    self.rs[n] = new_rs
            iteration += 1         

        print iteration
        # Plot objective function
        plt.scatter(x = np.arange(self.obj_output.shape[0]), y = self.obj_output)
        plt.show()
        print "DONE FITTING"
        return self.rs, self.means




    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.means

    # This should return the arrays for D images from each cluster that are representative of the clusters.
    def get_representative_images(self, D):
        print "FINDING REPRESENTATIVE IMAGES NAO"
        count = 0
        N = self.X.shape[0]
        img_array = np.zeros((self.K * D, 28, 28))
        for k in range(self.K):
            # Store distance and index
            distance = np.empty(0, dtype=[('x', float), ('y', int)])
            for n in range(N):
                if self.rs[n][k]:
                    tup = (np.linalg.norm(self.X[n] - self.means[k]), n)
                    distance = np.append(distance, np.array([tup], dtype=distance.dtype))
            distance.sort(order='x')

            for d in range(D):
                img_array[count] = self.X[distance[d][1]]
                count += 1

        return img_array

    # img_array should be a 2D (square) numpy array.
    # Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
    # However, we do ask that any images in your writeup be grayscale images, just as in this example.
    def create_image_from_array(self, img_array):
        print "GENERATING IMAGES NAO"
        for image in img_array:
            plt.figure()
            plt.imshow(image, cmap='Greys_r')
            plt.show()
        return

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.
K = 10
KMeansClassifier = KMeans(K=10, useKMeansPP=True)
start = time.time()
KMeansClassifier.fit(pics)
end = time.time()
KMeansClassifier.create_image_from_array(KMeansClassifier.get_mean_images())
KMeansClassifier.create_image_from_array(KMeansClassifier.get_representative_images(3))
print "TOOK: " + str(end-start) + " SECONDS."



