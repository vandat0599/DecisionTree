import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
X=iris.data
y=iris.target
#split dataset into training data and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=42)

def entropy(counts, n_samples):
    """
    Parameters:
    -----------
    counts: shape (n_classes): list number of samples in each class
    n_samples: number of data samples
    
    -----------
    return entropy 
    """
    #TODO
    entropy = 0
    for c in counts:
        entropy += (c/n_samples)*math.log2((c/n_samples))
    return -entropy

