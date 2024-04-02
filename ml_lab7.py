import numpy as np
from collections import Counter
from math import log2

class DecisionTree:
    def __init__(self):
        self.tree = None

    def _entropy(self, y):
        """
        Calculate the entropy of a given target variable.
        """
        class_counts = Counter(y)
        entropy = 0
        total_samples = len(y)
        for class_count in class_counts.values():
            p_class = class_count / total_samples
            entropy -= p_class * log2(p_class)
        return entropy

    def _information_gain(self, X, y, feature_index):
        """
        Calculate the information gain for a specific feature.
        """
        total_entropy = self._entropy(y)
        unique_values = set(X[:, feature_index])
        weighted_entropy = 0
        for value in unique_values:
            subset_indices = np.where(X[:, feature_index] == value)
            subset_entropy = self._entropy(y[subset_indices])
            weighted_entropy += (len(subset_indices[0]) / len(y)) * subset_entropy
        information_gain = total_entropy - weighted_entropy
        return information_gain

    def _find_best_split(self, X, y):
        """
        Find the best feature to split on based on maximum information gain.
        """
        num_features = X.shape[1]
        best_information_gain = -np.inf
        best_feature_index = None
        for i in range(num_features):
            information_gain = self._information_gain(X, y, i)
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature_index = i
        return best_feature_index

    def _split_dataset(self, X, y, feature_index, value):
        """
        Split the dataset based on a given feature and its value.
        """
        indices = np.where(X[:, feature_index] == value)
        return X[indices], y[indices]

    def _build_tree(self, X, y):
        """
        Recursively build the Decision Tree.
        """
        if len(set(y)) == 1:
            return y[0]  # Leaf node, return the class label
        
        if X.shape[1] == 0:
            return Counter(y).most_common(1)[0][0]  # No features left, return the majority class
        
        best_feature_index = self._find_best_split(X, y)
        best_feature_values = set(X[:, best_feature_index])
        sub_tree = {best_feature_index: {}}
        for value in best_feature_values:
            sub_X, sub_y = self._split_dataset(X, y, best_feature_index, value)
            sub_tree[best_feature_index][value] = self._build_tree(sub_X, sub_y)
        return sub_tree

    def fit(self, X, y):
        """
        Fit the Decision Tree to the training data.
        """
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        """
        Predict the class labels for input samples.
        """
        if self.tree is None:
            raise Exception("The model is not trained yet. Please call 'fit' method first.")
        predictions = []
        for sample in X:
            current_node = self.tree
            while isinstance(current_node, dict):
                feature_index = list(current_node.keys())[0]
                value = sample[feature_index]
                current_node = current_node[feature_index][value]
            predictions.append(current_node)
        return np.array(predictions)

def calculate_information_gain(tree, X, y):
    """
    Calculate the information gain for each feature.
    """
    total_entropy = tree._entropy(y)
    num_features = X.shape[1]
    information_gains = []
    for i in range(num_features):
        information_gain = tree._information_gain(X, y, i)
        information_gains.append(information_gain)
    return information_gains

def equal_width_binning(data, num_bins):
    """
    Perform equal width binning on continuous-valued data.
    """
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / num_bins
    bins = [min_val + i * bin_width for i in range(num_bins)]
    bins.append(max_val)
    return bins

def frequency_binning(data, num_bins):
    """
    Perform frequency binning on continuous-valued data.
    """
    sorted_data = sorted(data)
    bin_size = len(data) // num_bins
    bins = [sorted_data[i * bin_size] for i in range(num_bins)]
    bins.append(max(sorted_data))
    return bins

def binning(data, num_bins, method='equal_width'):
    """
    Perform binning on continuous-valued data using the specified method.
    """
    if method == 'equal_width':
        return equal_width_binning(data, num_bins)
    elif method == 'frequency':
        return frequency_binning(data, num_bins)
    else:
        raise ValueError("Invalid binning method. Supported methods are 'equal_width' and 'frequency'.")

def preprocess_data(tree, X, y, binning_method='equal_width', num_bins=5):
    """
    Preprocess the data by binning continuous-valued features and calculating information gain.
    """
    categorical_X = []
    for i in range(X.shape[1]):
        if len(set(X[:, i])) < num_bins:
            categorical_X.append(X[:, i])
        else:
            bins = binning(X[:, i], num_bins, binning_method)
            binned_feature = np.digitize(X[:, i], bins)
            categorical_X.append(binned_feature)
    categorical_X = np.array(categorical_X).T
    information_gains = calculate_information_gain(tree, categorical_X, y)
    return categorical_X, information_gains

# Load your dataset here
def load_dataset():
    """
    Load your dataset here.
    """
    # Replace this with code to load your dataset
    X = np.array([
        [20, 6, 2],
        [16, 3, 6],
        [27, 6, 2],
        [19, 1, 2],
        [24, 4, 2],
        [22, 1, 5],
        [15, 4, 2],
        [18, 4, 2],
        [21, 1, 4],
        [16, 2, 4]
    ])
    y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    return X, y

# Example usage:
# Load dataset
X, y = load_dataset()

# Create an instance of the DecisionTree class
tree = DecisionTree()

# Preprocess the data
X_categorical, information_gains = preprocess_data(tree, X, y)

# Create and train the Decision Tree model
tree.fit(X_categorical, y)

# Assuming X_test is your test data
X_test = np.array([
    [20, 6, 2],
    [16, 3, 6],
    [27, 6, 2],
    [19, 1, 2],
    [24, 4, 2]
])

# Make predictions
predictions = tree.predict(X_test)
print("Predictions:", predictions)
