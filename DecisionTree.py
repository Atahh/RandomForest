'''
Authors: Atah and Jacobo
CS429/529
Project 1: Random Forest
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score

import math

# Read in Data
cols = pd.read_csv('train.csv', nrows=0).columns
dataSet = pd.read_csv('train.csv', usecols=cols[1:])
df = pd.DataFrame(dataSet)
# Alpha Value for Chi2 Calculations
alpha = 0.01

def preprocess_data_replace_with_column_avg(df):
    for col in df.columns:
        # Convert 'NotFound' to NaN, then attempt to convert the column to numeric, coercing errors to NaN        
        # If after conversion the column is numeric (and not just an empty series of NaNs)
        if pd.to_numeric(df[col],errors='coerce').notna().any():
            numeric_col = pd.to_numeric(df[col].replace('NotFound', np.nan))

            # Calculate mean, but ensure there's at least one non-NaN value
            if numeric_col.notna().sum() > 0:
                valid_mean = numeric_col.mean()
                # Replace NaN with the calculated mean
                df[col] = numeric_col.fillna(valid_mean)
            else:
                # Optional: Handle columns that are entirely NaN after conversion, if necessary
                df[col] = numeric_col.fillna(0)  # or any other default value
        else:
            # Handle non-numeric columns or leave as is
            pass

    return df

# Preprocess the data to handle 'NotFound'
df = preprocess_data_replace_with_column_avg(df)



# Calculate entropy
def calculate_entropy(data, target_column):
    total_rows = len(data)
    target_values = data[target_column].unique()
 
    entropy = 0
    for value in target_values:
        # Calculate the proportion of instances with the current value
        value_count = len(data[data[target_column] == value])
        proportion = value_count / total_rows
        entropy -= proportion * math.log2(proportion)
 
    return entropy

# Calculate Misclassification Error
def calculate_misclassification_error(dataset, target_column):
    total_rows = len(dataset) 
    
    prob = {}
    for value in (dataset[target_column]):
        #print(value)
        if not value in prob:
            prob[value] = 1
        else:
            prob[value] += 1       
    
    probabilities = [count / total_rows for count in prob.values()]
    #print(probabilities)
    misclassification_error = 1 - max(probabilities)
    return misclassification_error

# Calculate Gini Index
def calculate_gini_index(dataset, target_column):
    total_rows = len(dataset) 
    
    prob = {}
    for value in (dataset[target_column]):
        #print(value)
        if not value in prob:
            prob[value] = 1
        else:
            prob[value] += 1          
    
    probabilities = [count / total_rows for count in prob.values()]
    #print(probabilities)
    gini_index = 1 - sum([p**2 for p in probabilities])
    return gini_index



#Entropy Value to use for Info Gain
#entropy_outcome = calculate_entropy(df,'isFraud')
#gini_outcome = calculate_gini_index(df,'isFraud')
#miss_outcome = calculate_misclassification_error(df,'isFraud')


def calculate_information_gain(data, feature, target_column, criterium=calculate_entropy):
 
    # Calculate weighted average entropy for the feature
    unique_values = data[feature].unique()
    weighted_entropy = 0
 
    for value in unique_values:
        subset = data[data[feature] == value]
        proportion = len(subset) / len(data)
        weighted_entropy += proportion * criterium(subset, target_column)
 
    # Calculate information gain
    information_gain = criterium(data, 'isFraud') - weighted_entropy
 
    return information_gain



#for column in dataSet.columns[:-1]:
#    entropy = calculate_entropy(df,column)
#    infoGain = calculate_information_gain(df,column,'isFraud')
#    print(f"{column} - Entropy: {entropy:.3f}, Info Gain {infoGain:.3f}")
    
    
class TreeNode:
    def __init__(self, attribute_idx, decision=None):
        self.attribute_idx = attribute_idx
        self.children = {}
        self.decision = decision

    def add_child(self, value, node):
        self.children[value] = node

    def remove_child(self, value):
        del self.children[value]

    def get_children(self):
        return self.children.values()
    
    def get_child(self, value):
        return self.children[value]
    

    def print_tree(self, depth=0):
        if self.attribute_idx is not None:
            print('  ' * depth, 'Attribute: ', self.attribute_idx)
            for value, child in self.children.items():
                print('  ' * (depth + 1), 'val: ', value)
                child.print_tree(depth + 2)
        else:
            print('  ' * depth, 'decision: ', self.decision)



class DecisionTree:
    def __init__(self, target_column,max_depth=None):
        self.root = None
        self.target_column = target_column
        self.max_depth = max_depth

    def fit(self, data, max_features=None):
        if max_features is None:
            max_features = len(data.columns) - 1  # Exclude target column
        self.root = self.build_tree(data, [], max_features,)

    # Function to discretize numeric features into categorical bins, useful for handling continuous data.
    def discretize_numeric_feature(self, data, attribute, bins=10):
        data_copy = data.copy()
        if pd.api.types.is_numeric_dtype(data[attribute]):
            # Ensure the column does not consist entirely of NaN values
            if not data_copy[attribute].isna().all():
                # Calculate the number of unique values to avoid issues with bin edges
                unique_values = data_copy[attribute].nunique(dropna=True)
                # Only attempt to bin if there are enough unique values to form bins
                if unique_values > 1:
                    try:
                        data_copy[attribute] = pd.cut(data_copy[attribute], bins, labels=range(bins), duplicates='drop')
                    except ValueError as e:
                        print(f"Error discretizing {attribute}: {e}")
                else:
                    # Handle the case where there aren't enough unique values for binning
                    data_copy[attribute] = 0 
            else:
                # If the column is entirely NaN
                data_copy[attribute].fillna(0, inplace=True)
        return data_copy


    def build_tree(self, data, ignored_attributes=[], max_features=None,current_depth=0):
        # Base case 1: If all rows in the dataset have the same label, return a leaf node with that label.
        if len(data[self.target_column].unique()) == 1:
            return TreeNode(None, decision=data[self.target_column].iloc[0])

        # Base case 2: If all features have been considered (or if no features are left), 
        # return a leaf node with the most common target value in the dataset.
        if len(data.columns) - 1 == len(ignored_attributes):
            most_common_target = data[self.target_column].mode()[0]
            return TreeNode(None, decision=most_common_target)
        
        # Check if max depth has been reached
        if self.max_depth is not None and current_depth >= self.max_depth:
            most_common_target = data[self.target_column].mode()[0]
            return TreeNode(None, decision=most_common_target)

        # Selecting features to consider for splitting: Exclude any ignored attributes and the target column.
        available_features = [col for col in data.columns if col not in ignored_attributes + [self.target_column]]
        
        # Determine the number of features to consider based on the 'max_features' parameter.
        # 'max_features' controls feature sampling for splits, contributing to model robustness and preventing overfitting.
        n_features_to_consider = min(max_features, len(available_features)) if max_features else len(available_features)

        if n_features_to_consider > 0:
            # Randomly select features to consider for the split to add randomness to the tree construction, 
            # which is especially useful in a Random Forest setup.
            features_to_consider = np.random.choice(available_features, n_features_to_consider, replace=False)

            best_attribute, best_info_gain, p_value = None, -1, float('inf')
            for attribute in features_to_consider:
                # Discretize the feature if necessary and reset index to avoid issues with pandas operations that 
                # assume unique indices.
                discretized_data = self.discretize_numeric_feature(data.copy(), attribute)
                discretized_data.reset_index(drop=True, inplace=True)
                
                # Calculate information gain for the attribute and use a contingency table for chi-square test to 
                # determine if the split improves the model significantly.
                info_gain = calculate_information_gain(discretized_data, attribute, self.target_column)
                target_series = data[self.target_column].reset_index(drop=True)
                contingency_table = pd.crosstab(discretized_data[attribute], target_series)
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                
                # Update the best attribute if the current one has a higher information gain and passes the chi-square test.
                if info_gain > best_info_gain and p < alpha:
                    best_info_gain = info_gain
                    best_attribute = attribute
                    p_value = p
                    
            #print(f"P Value: {p_value}")

            if best_attribute:
                # If a best attribute is found, create a node for it and recursively build the tree for each value of this attribute.
                node = TreeNode(best_attribute)
                for value in data[best_attribute].unique():
                    subset = data[data[best_attribute] == value]
                    if not subset.empty:
                        node.add_child(value, self.build_tree(subset, ignored_attributes + [best_attribute], max_features,current_depth+1))
                return node
            else:
                # If no attribute significantly improves the model, return a leaf node with the most common target value.
                most_common_target = data[self.target_column].mode()[0]
                return TreeNode(None, decision=most_common_target)
        else:
            # If no features are available to consider, return a leaf node with the most common target value.
            most_common_target = data[self.target_column].mode()[0]
            return TreeNode(None, decision=most_common_target)


    def print_tree(self):
        if self.root:
            self.root.print_tree()

    def predict(self, X):
        # Ensure X is iterated row by row
        return [self._predict_single(row) for _, row in X.iterrows()]

    def _predict_single(self, row, node=None):
        if node is None:
            node = self.root
        if node.decision is not None:
            return int(node.decision)  # Ensure decision is cast to int
        # Recurse down the tree based on the value of the splitting attribute in the current row
        attribute_value = row[node.attribute_idx]
        if attribute_value in node.children:
            return self._predict_single(row, node.children[attribute_value])
        else:
            # Handle missing or unseen attribute values
            return 1  # False Positive
                



class RandomForest:
    def __init__(self, target_column, n_trees=10, max_features='max', bag_size=1.0,max_depth = None):
        self.target_column = target_column
        self.n_trees = n_trees
        self.max_features = max_features
        self.bag_size = bag_size
        self.max_depth = max_depth
        self.trees = []

    def fit(self, data):
        n_rows, n_cols = data.shape
        n_features = n_cols - 1  # Excluding target column
        
        # Determine the number of features to consider at each split
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif self.max_features == 'max':
            max_features = n_features  # Use all features
        
        for _ in range(self.n_trees):
            # Bagging
            sample = resample(data, n_samples=int(self.bag_size * len(data)))
            # Initialize a new decision tree with a subset of features
            tree = DecisionTree(self.target_column, max_depth=self.max_depth)
            tree.fit(sample, max_features=max_features)
            self.trees.append(tree)

    def predict(self, X):
        # Initialize a matrix to hold predictions from each tree
        predictions = np.zeros((len(X), self.n_trees), dtype=np.int64)  

        # Collect predictions from each tree
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)

        # Determine the majority vote for each instance
        majority_votes = [np.bincount(predictions[i]).argmax() for i in range(len(X))]

        return majority_votes
    


'''
Testing Code
'''
# Split training data into features and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,random_state=45)

# Combine X_train and y_train for random forest fitting
train_data = pd.concat([X_train, y_train], axis=1)

# Initialize and fit the RandomForest model
random_forest = RandomForest(target_column='isFraud', n_trees=10,max_features='max',max_depth=10)
random_forest.fit(train_data)

# Prepare the validation data (similar to test data preparation)
# This step simulates preparing actual test data, using validation data here
val_data = X_val.copy()
val_data['isFraud'] = y_val  # Add 'isFraud' back for evaluation purposes

# Make predictions on the validation set to evaluate the model
predictions_val = random_forest.predict(val_data.drop('isFraud', axis=1))

# Calculate balanced accuracy on the validation set
balanced_acc = balanced_accuracy_score(y_val, predictions_val)

print(f"Balanced Accuracy on Validation Set: {balanced_acc}")

# Load the actual test dataset
test_df = pd.read_csv('test.csv')
test_df_processed = preprocess_data_replace_with_column_avg(test_df)


# Prepare the test dataset features for prediction
X_test_actual = test_df_processed.drop('TransactionID', axis=1)

# Make predictions on the actual test dataset
predictions_test = random_forest.predict(X_test_actual)

# Create a submission DataFrame with 'TransactionID' and the predictions
submission_df = pd.DataFrame({
    'TransactionID': test_df['TransactionID'],
    'isFraud': predictions_test
})

# Submission Stuff

# Write the submission DataFrame to a CSV file for submission
submission_df.to_csv('submission22.csv', index=False)

print("Submission file 'submission.csv' created.")
