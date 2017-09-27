#  todo:
#     check algorithm  (is that identification really correct?)
#     refactor

import sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heapq

from PIL import Image

# returns difference between two vectors
def euclidean_dist(x, y):
    diff_vect = x - y
    return np.linalg.norm(diff_vect)

k = int(input("please type a k for how many nearest neighbors you want to compute"))
num_training_pts = int(input("There are 10,000 pictures of numbers.\n  How many would you like to use as training data?"))


data = sio.loadmat('ML_hw1data.mat')        # dictionary
training_pixels = data['X'][:num_training_pts]   #          10000x784
training_labels = data['Y'][:num_training_pts]   #          10000x  1
test_pixels     = data['X'][num_training_pts:]   # use more test data later
test_labels     = data['Y'][num_training_pts:]

# nbrs = Nearest




correct_wrong_list = [0,0]
RIGHT = 0
WRONG = 1

#   max heap stores the worst of the closest Euclidean distances

#  go through each test picture and match it against each training pic
for test_idx in range(0, len(test_pixels)):

    knn = []   # [euc_dist, 'idx']

    #   finding k nearest neighbors
    for i in range(0, 100):
        img = training_pixels[i]
        curr_dist = euclidean_dist(img, test_pixels[test_idx])
        if (len(knn) < k):
            heapq.heappush(knn, (curr_dist, str(i)))
            heapq._heapify_max(knn)
        elif (curr_dist < knn[0][0]):
            heapq.heappop(knn)
            heapq.heappush(knn, (curr_dist, str(i)))
            heapq._heapify_max(knn)


    #    counting up labels and finding most likely one
    counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for pair in knn:
        idx = int(pair[1])
        label = training_labels[idx][0]
        counts[label] += 1

    label_guess = 0
    prev_max_num_guesses = 0   #  rename
    for i in range(0, 10):
        if (counts[i] > prev_max_num_guesses):
            label_guess = i
    actual_label = test_labels[test_idx][0]
    if (actual_label == label_guess):
        correct_wrong_list[RIGHT] += 1
    else : 
        correct_wrong_list[WRONG] += 1

    print("   test image ", end='')
    print((num_training_pts + test_idx), end='')
    print(" was identified as ", end='')
    print(label_guess)
    print("     actual label was ", end='')
    print(test_labels[test_idx][0])


print("    k was ", end='')
print(k)
print("    and we used ", end='')
print(num_training_pts, end='')
print(" training data points")
print("\n\n\n\n   Correct vs. wrong answers:")
print(correct_wrong_list)
print("\n\n\n\n\n    Percentage correct:    ")
print(   (correct_wrong_list[0] +0.0)   /   correct_wrong_list[1])





# ________________________________
#
#        Data collection
# ________________________________

#  This is for          k = 5  
#              n_training = 100
#          right  wrong
#  we got [1852, 8048]

#  This is for          k = 5  
#              n_training = 1000
#          right  wrong
#  we got [1694, 7306]

#  This is for          k = 1  
#              n_training = 100
#          right  wrong
#  we got [1609, 8291]

#     print(type(Y))  #  ndarray

#    how to reshape an image to size it properly:    a = X[idx,:].reshape((28,28))

