import numpy as np
import pandas as panda

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def gini_index(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    gini = 1 - np.sum([(counts[i]/np.sum(counts))**2 for i in range(len(elements))])
    return gini

def misclassification_error(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    misclassification = 1 - np.max(counts)/np.sum(counts)
    return misclassification

def information_gain(data, split_attribute_name, target_name, criterion="entropy"):
    # Calculate the entropy or gini of the total dataset
    total_entropy = 0
    if criterion == "entropy":
        total_entropy = entropy(data[target_name])
    elif criterion == "gini":
        total_entropy = gini_index(data[target_name])
    elif criterion == "misclassification":
        total_entropy = misclassification_error(data[target_name])
    
    # Calculate the values and the corresponding counts for the split attribute 
    vals, counts= np.unique(data[split_attribute_name], return_counts=True)
    
    # Calculate the weighted entropy
    Weighted_Entropy = sum([(counts[i]/np.sum(counts)) * 
                            (entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) if criterion == "entropy" 
                             else gini_index(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) if criterion == "gini"
                             else misclassification_error(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]))
                            for i in range(len(vals))])
    
    # Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain