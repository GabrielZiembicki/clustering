#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import sys
import matplotlib.pyplot as plt


n_clusters = 5

def generateX(n_clusters=n_clusters):
    X = []
    for i in range(1,n_clusters+1):
        mean = [3*i, 3*i]
        cov = [[i, 0],
               [0, i]]
        X.append(np.random.multivariate_normal(mean, cov, 1000))    
    X = np.vstack(X)
    return X

X = np.genfromtxt(sys.argv[1], delimiter = ",")
#X = generateX()

def make_plot(c, mu):
    plt.clf()
    plt.scatter(X[:,0], X[:,1], c=c)
    plt.scatter(mu[:,0], mu[:,1], marker='+', s=100, c='r')
    plt.show()
    
    
def make_plotGMM(mu, w):
    plt.clf()
    plt.scatter(X[:,0], X[:,1], facecolors=w)
    plt.scatter(mu[:,0], mu[:,1], marker='+', s=100, c='r')
    plt.show()

def initialize_clusters(X):
    idx = np.random.randint(low=0, high=len(X), size=n_clusters)
    mu = np.array([X[index] for index in idx])
    G = [[[1,0], [0,1]] for i in range(n_clusters)]
    pi = [1/n_clusters]*n_clusters
    return np.array(mu), G, pi

def Lpenalty(c, *args):
    c = np.array([int(c_) for c_ in c])
    X_mu = np.array([mu[c[i]] for i in range(len(X))])
    distance2 = (np.subtract(X, X_mu))**2
    L = np.sum(distance2)
    return L
    
def KMeans(X):
	#perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively
    mu, _, _ = initialize_clusters(X)
    c = np.random.randint(0, n_clusters, len(X))
    for i in range(10):
        for xi in range(len(X)):
            dist = (np.linalg.norm(X[xi] - mu, axis=1))**2
            c[xi] = np.argmin(dist)      
        for ki in range(n_clusters):
            mu[ki] = np.mean(X[c==ki])
#    make_plot(c, mu)    
        filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, mu, delimiter=",")
KMeans(X)   
  
def EMGMM(X):
    w = np.zeros((len(X), n_clusters))
    n = np.zeros(n_clusters)
    mu, G, pi = initialize_clusters(X)
    for i in range(10):
        for xi in range(X.shape[0]):
            for k in range(n_clusters):
                w[xi, k] = pi[k] * sp.stats.multivariate_normal.pdf(X[xi,:], mean=mu[k], cov=G[k])
            w[xi, :] = w[xi, :] / np.sum(w[xi, :])
        
        for k in range(n_clusters):
            n[k] = np.sum(w[:, k], axis=0)
            mu[k] = 1/n[k] * np.dot(w[:, k], X)
            mu[k] = np.average(X, axis=0, weights=w[:, k])
            pi[k] = n[k]/len(X)
#            G[k] = np.average((X-mu[k])**2, axis=0, weights=w[:, k])
            G[k] = np.cov(X.T, ddof=0, aweights=w[:,k])
#            G[k] = 1/n[k] * np.dot(w[:,k], np.dot(X-mu[k], (X-mu[k]).T))
#    make_plotGMM(mu, w)  

       
        
        filename = "pi-" + str(i+1) + ".csv" 
        np.savetxt(filename, pi, delimiter=",") 
        filename = "mu-" + str(i+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
        
        for j in range(k): #k is the number of clusters 
            filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, G[j], delimiter=",")
    
EMGMM(X)    
    
    


