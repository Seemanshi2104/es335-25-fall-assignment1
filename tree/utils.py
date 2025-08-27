"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    Will return True if the given series is real
    """
    # If dtype is category, it's discrete; otherwise considered real
    return not y.dtype =="category"


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    Entropy= - Σ p*log2(p)
    """
    p =Y.value_counts(normalize=True) # probability distribution
    p[p == 0.0] = 1.0 # avoid log(0)
    S =-np.sum(p*np.log2(p))
    return S


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    Gini = 1-Σ(p^2)
    """
    p =Y.value_counts(normalize=True)
    return 1 - np.sum(p*p)

def mse(Y: pd.Series) -> float:
    """
    Function to calculate mse of data
    (used for regression)
    """
    return np.mean((Y-np.mean(Y))**2)

def get_criterion_function(criterion):
    """
    Utility function to return the appropriate impurity function
    depending on the given criterion
    """
    fn = None
    if criterion =='information_gain':
        fn = entropy
    elif criterion =='gini_index':
        fn = gini_index
    elif criterion =='mse':
        fn = mse
    else:
        raise NotImplementedError("Criterion must be one of 'information_gain', 'gini_index', or 'mse'")
    return fn

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    Y: output values
    attr: feature column to split upon
    criterion: impurity measure

    Returns:
        best_gain: maximum information gain
        opt_split: best split value for the attribute
    """
    fn =None
    if criterion =='information_gain':
        fn =entropy
    elif criterion =='gini_index':
        fn = gini_index
    elif criterion =='mse':
        fn =mse
    else:
        raise NotImplementedError("Criterion must be one of 'information_gain', 'gini_index', or 'mse'")
    
    prev_info = fn(Y)
    attr_values = attr.unique()
    attr_values.sort()
    best_gain = 0
    opt_split = None
    
    for value in attr_values:
        left = Y[attr<= value]
        right = Y[attr> value]
        current_info = fn(left) *len(left) /len(Y) + fn(right) * len(right) / len(Y)
        info_gain = prev_info - current_info
        if info_gain > best_gain:
            best_gain = info_gain
            opt_split = value
    
    assert (opt_split is not None or best_gain == 0)    
    return best_gain, opt_split


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    """
    Function to find the optimal attribute to split upon.

    X: feature dataframe
    y: target series
    criterion: impurity measure (entropy, gini, mse)
    features: list of attributes to consider

    Returns:
        best_attr: best attribute to split on
        best_information_gain: corresponding information gain
    """
    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_information_gain = -np.inf
    best_attr = None

    for feature in features:
        current_information_gain = information_gain(y, X[feature], criterion)
        if current_information_gain > best_information_gain:
            best_information_gain = current_information_gain
            best_attr = feature

    return best_attr, best_information_gain


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    assert y.size == X.shape[0]
    X_left, y_left, X_right, y_right = None, None, None, None

    X_left, y_left = X[X[attribute] <= value], y[X[attribute] <= value]
    X_right, y_right = X[X[attribute] > value], y[X[attribute] > value]

    return X_left, y_left, X_right, y_right 
