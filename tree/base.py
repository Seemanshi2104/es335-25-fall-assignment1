"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
"""
Decision Tree Base Implementation
Handles:
1. Discrete Input, Discrete Output
2. Real Input, Real Output
3. Real Input, Discrete Output
4. Discrete Input, Real Output
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

@dataclass
class Node:
    children:list
    criterion:str
    depth:int
    samples_x:pd.DataFrame
    samples_y:pd.Series
    split_feature: str
    
    
    def __init__(self,criterion,depth):
        self.children =[]
        self.criterion =criterion
        self.depth =depth
        self.samples_x =None
        self.samples_y =None
        self.split_feature =None
        self.split_value =None
        
    def split(self, max_depth,eps):
        """
        Recursively split the node until max_depth or 
        impurity is below threshold eps.
        """
        assert (self.samples_x is not None) and (self.samples_y is not None)

        #Base cases
        if self.depth ==max_depth:
            return
        
        if get_criterion_function(self.criterion)(self.samples_y) <=eps:
            return

        best_gain = -np.inf
        opt_split =None
        opt_feature =None

        for feature in self.samples_x.columns:
            info_gain, current_opt_split =information_gain(self.samples_y,self.samples_x[feature], self.criterion)
            
            if info_gain > best_gain:
                opt_split =current_opt_split
                best_gain = info_gain
                opt_feature = feature
        
        if best_gain == 0:
            return
                
        assert (opt_split is not None) and (opt_feature is not None)
        
        self.split_feature = opt_feature
        self.split_value = opt_split
        
        left_child_mask = self.samples_x[opt_feature] <= opt_split
        right_child_mask = self.samples_x[opt_feature] > opt_split
        
        self.children = [Node(self.criterion, self.depth + 1), Node(self.criterion, self.depth + 1)]
        
        self.children[0].add_samples(self.samples_x[left_child_mask], self.samples_y[left_child_mask])
        self.children[1].add_samples(self.samples_x[right_child_mask], self.samples_y[right_child_mask])
        
        self.children[0].split(max_depth, eps)
        self.children[1].split(max_depth, eps)
    
    def add_samples(self, samples_x, samples_y):
        self.samples_x = samples_x
        self.samples_y = samples_y
    
    def get_value(self):
        """
        Return prediction at this node (leaf).
        - If classification: return most frequent class (mode)
        - If regression: return mean of values
        """
        if not check_ifreal(self.samples_y):
            return self.samples_y.mode()[0]
        else:
            return self.samples_y.mean()
        
    def predict(self, X: pd.DataFrame):
        if len(self.children) ==0:
            return np.array([self.get_value()] *X.shape[0])
    
        res = np.empty(X.shape[0], dtype=object)
        left_mask = X[self.split_feature] <= self.split_value
        right_mask = ~left_mask
        res[left_mask] = self.children[0].predict(X[left_mask])
        res[right_mask] = self.children[1].predict(X[right_mask])
        return res

        
    def plot(self):
        if len(self.children) ==0:
            if not check_ifreal(self.samples_y):
                print(f"Class {str(self.get_value())},{self.samples_y.to_numpy()}")
            else:
                print(f"Value {str(self.get_value())},{self.samples_y.to_numpy()}")
        else:
            print(f"?{self.split_feature} <= {self.split_value}")
            print(("  " * (2 * self.depth + 1)) + "Yes: ",end='')
            self.children[0].plot()
            print(("  " * (2 * self.depth + 1)) + "No: ",end='')
            self.children[1].plot()
        

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index","mse"] 
    max_depth: int  # The maximum depth the tree can grow to
    root: Node
    eps: float

    def __init__(self, criterion, max_depth=5, eps=1e-7):
        self.criterion =criterion
        self.max_depth =max_depth
        self.root = Node(criterion, 0)
        self.eps = eps

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        X = one_hot_encoding(X) # handle discrete input 
        # If you wish your code can have cases for different types of input and output data (discrete, real)
        self.root.add_samples(X, y)
        self.root.split(self.max_depth, self.eps)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = one_hot_encoding(X)
        preds = self.root.predict(X)
        return pd.Series(preds).astype(self.root.samples_y.dtype)

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.root.plot()