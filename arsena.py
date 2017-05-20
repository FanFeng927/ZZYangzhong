__author__  ='faithefeng'
'''
     This file includes many user-defined functions. And for brevity and reusage,
     these function are saved in a standalone file.    
'''

import pandas as pd
import sklearn
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy.random import random_sample
from math import sqrt, log
# returns series of random values sampled between min and max values of passed col
def mean(l):
    return sum(l)/len(l)
def get_rand_data(col):
    rng = col.max() - col.min()
    return pd.Series(random_sample(len(col))*rng + col.min())

def iter_kmeans(df, n_clusters, num_iters=5):
    rng =  range(1, num_iters + 1)
    vals = pd.Series(index=rng)
    for i in rng:
        k = KMeans(n_clusters=n_clusters, n_init=3)
        k.fit(df)
        #print("Ref k: %s" % k.get_params()['n_clusters'])
        vals[i] = k.inertia_
    return vals

def gap_statistic(df, max_k=10):
    gaps = pd.Series(index = range(1, max_k + 1))
    sks =  pd.Series(index = range(1, max_k + 1))
    wks =  pd.Series(index = range(1, max_k + 1))
    for k in range(1, max_k + 1):
        km_act = KMeans(n_clusters=k, n_init=3)
        km_act.fit(df)

		# get ref dataset
        n_iters = 10
        ref = df.apply(get_rand_data)
        ref_inertia = iter_kmeans(ref, n_clusters=k,num_iters = n_iters)
        
        w_mean = mean([log(ele) for ele in ref_inertia]) 
        gap = w_mean - log(km_act.inertia_)
        std = mean([(log(ele)-w_mean)**2 for ele in ref_inertia])**(1/2)
        sk = std*(1/n_iters+1)**(1/2)
        
        gaps[k] = gap
        sks[k] = sk
        wks[k] = log(km_act.inertia_)
    return gaps,sks,wks
def plot_gaps(gaps,sks):
    plt.plot(np.arange(len(gaps)),gaps,'-o',color = 'b')
    plt.plot(pd.DataFrame([np.arange(len(gaps)),np.arange(len(gaps))]),pd.DataFrame([gaps+sks,gaps-sks]),'-',color = 'b')
    plt.plot(pd.DataFrame([np.arange(len(gaps))-0.2,np.arange(len(gaps))+0.2]),pd.DataFrame([gaps+sks,gaps+sks]),'-',color = 'b')
    plt.plot(pd.DataFrame([np.arange(len(gaps))-0.2,np.arange(len(gaps))+0.2]),pd.DataFrame([gaps-sks,gaps-sks]),'-',color = 'b')