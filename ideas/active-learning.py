#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 22:17:35 2018

@author: gabriel
"""









# -*- coding: utf-8 -*-
"""
Spyder                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

This is a temporary script file.
"""

import numpy as np
import sys

lambda_input = 0
sigma2_input = 1.2
#X_train = np.genfromtxt('/media/gabriel/Nowy/Learning/edX Machine Learning/Project/X_train.csv', delimiter = ",")
#y_train = np.genfromtxt('/media/gabriel/Nowy/Learning/edX Machine Learning/Project/y_train.csv')
X_train = np.array([[1,2,4,5,6,5,7,8,7,9], [2,4,6,8,7,11,23,14,15,17]]).T
y_train= np.array([1,2,3,4,5,6,7,8,9,10])

X_test = np.array([[0,2,3,4],[2.5,3,5,7]]).T

X_test = np.random.randint(20, size=(200, 2))

#lambda_input = int(sys.argv[1])
#sigma2_input = float(sys.argv[2])
#X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
#y_train = np.genfromtxt(sys.argv[4])
#X_test = np.genfromtxt(sys.argv[5], delimiter = ",")



## Solution for Part 1
def part1(X_train, y_train, lambda_input):
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    lb = np.identity(lambda_input)
    XT = np.transpose(X_train)
    XTX = XT.dot(X_train)
    XTX = XT.dot(X_train)
    lb = np.identity(len(XTX)) * lambda_input
    wRR = np.linalg.inv(lb + XTX)
    wRR = wRR.dot(XT)
    wRR = wRR.dot(y_train)
    return wRR

wRR = part1(X_train, y_train, lambda_input)  # Assuming wRR is returned from the function
#np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2(X_test, sigma2_input, lambda_input):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    idx = []
    lb = np.identity(len(X_test.T)) * lambda_input
    E = (lb + (sigma2_input**(-1) * X_test.T).dot(X_test))
    E = np.linalg.inv(E)
    for k in range(0, X_test.shape[0]):
        #print(E)
        entropy = []
        for i in range(X_test.shape[0]):
            x0 = X_test[i,:]
            ent_dif = sigma2_input + (x0.dot(E)).dot(x0)
            entropy.append((i, ent_dif))
        entropy_sorted = sorted(entropy, key=lambda t: t[1], reverse=True)
        max_ent_idx = entropy_sorted[0][0]

        z=0
        while max_ent_idx+1 in idx:
            z+=1
            max_ent_idx = entropy_sorted[z][0]
        #print(entropy)
        idx.append(max_ent_idx+1)
        x0 = X_test[max_ent_idx,:]
        E = np.linalg.inv(np.linalg.inv(E) + (sigma2_input**(-1) * np.dot(x0, x0.T)))
        
    return np.array(idx)


active = part2(X_test, sigma2_input, lambda_input) # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", [active], delimiter=",") # write output to file                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

