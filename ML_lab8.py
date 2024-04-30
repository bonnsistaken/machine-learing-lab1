import pandas as pd
import numpy as np

def calculate_entropy(y):
    """
    Calculate entropy for a given target variable.
    """
    classes = np.unique(y)
    entropy = 0
    total_samples = len(y)
    for cls in classes:
        p_cls = np.sum(y == cls) / total_samples
        entropy -= p_cls * np.log2(p_cls)
    return entropy

def calculate_information_gain(X, y, feature):
    """
    Calculate information gain for a given feature.
    """
    # Calculate entropy of the entire dataset
    total_entropy = calculate_entropy(y)

    # Calculate entropy for each unique value of the feature
    unique_values = np.unique(X[feature])
    entropy_feature = 0
    total_samples = len(y)
    for value in unique_values:
        subset_indices = X[feature] == value
        subset_entropy = calculate_entropy(y[subset_indices])
        entropy_feature += (np.sum(subset_indices) / total_samples) * subset_entropy

    # Calculate information gain
    information_gain = total_entropy - entropy_feature
    return information_gain

def find_root_node(X, y):
    """
    Find the root node feature based on information gain.
    """
    features = X.columns
    max_information_gain = -1
    best_feature = None
    for feature in features:
        information_gain = calculate_information_gain(X, y, feature)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_feature = feature
    return best_feature

# Example usage:
if __name__ == "__main__":
    # Assuming data is loaded from a CSV file
    data = pd.read_csv("/content/drive/MyDrive/Bangli-P10_gabor.csv", nrows=1048571)

    # Assuming the second column is the target variable
    y = data.iloc[:, 2]     # Target
    X = data.iloc[:, 3:30]  # Features

    # Find the root node feature
    root_node = find_root_node(X, y)
    print("Root Node Feature:", root_node)


    #secondd
import pandas as pd
import numpy as np

def calculate_entropy(y):
    """
    Calculate entropy for a given target variable.
    """
    classes = np.unique(y)
    entropy = 0
    total_samples = len(y)
    for cls in classes:
        p_cls = np.sum(y == cls) / total_samples
        entropy -= p_cls * np.log2(p_cls)
    return entropy

def calculate_information_gain(X, y, feature):
    """
    Calculate information gain for a given feature.
    """
    # Calculate entropy of the entire dataset
    total_entropy = calculate_entropy(y)

    # Calculate entropy for each unique value of the feature
    unique_values = np.unique(X[feature])
    entropy_feature = 0
    total_samples = len(y)
    for value in unique_values:
        subset_indices = X[feature] == value
        subset_entropy = calculate_entropy(y[subset_indices])
        entropy_feature += (np.sum(subset_indices) / total_samples) * subset_entropy

    # Calculate information gain
    information_gain = total_entropy - entropy_feature
    return information_gain

def find_root_node(X, y):
    """
    Find the root node feature based on information gain.
    """
    features = X.columns
    max_information_gain = -1
    best_feature = None
    for feature in features:
        information_gain = calculate_information_gain(X, y, feature)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_feature = feature
    return best_feature

def equal_width_binning(feature, num_bins):
    """
    Perform equal width binning for a continuous feature.
    """
    bins = pd.cut(feature, bins=num_bins, labels=False)
    return bins

def frequency_binning(feature, num_bins):
    """
    Perform frequency binning for a continuous feature.
    """
    bins = pd.qcut(feature, q=num_bins, labels=False, duplicates='drop')
    return bins

# Example usage:
if __name__ == "__main__":
    # Assuming data is loaded from a CSV file
    data = pd.read_csv("/content/drive/MyDrive/Bangli-P10_gabor.csv", nrows=1048571)

    # Assuming the second column is the target variable
    y = data.iloc[:, 2]     # Target
    X = data.iloc[:, 3:28].copy()  # Features (make a copy to avoid SettingWithCopyWarning)

    # Binning the continuous features
    for column in X.columns:
        if X[column].dtype == 'float64' or X[column].dtype == 'int64':
            # Using equal width binning with 5 bins as default
            X[column] = equal_width_binning(X[column], num_bins=5)

    # Find the root node feature
    root_node = find_root_node(X, y)
    print("Root Node Feature:", root_node)

    # Display the binned features
    print("\nBinned Features:")
    print(X.head())

#third
import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def calculate_entropy(self, y):
        """
        Calculate entropy for a given target variable.
        """
        classes = np.unique(y)
        entropy = 0
        total_samples = len(y)
        for cls in classes:
            p_cls = np.sum(y == cls) / total_samples
            entropy -= p_cls * np.log2(p_cls)
        return entropy

    def calculate_information_gain(self, X, y, feature):
        """
        Calculate information gain for a given feature.
        """
        # Calculate entropy of the entire dataset
        total_entropy = self.calculate_entropy(y)

        # Calculate entropy for each unique value of the feature
        unique_values = np.unique(X[feature])
        entropy_feature = 0
        total_samples = len(y)
        for value in unique_values:
            subset_indices = X[feature] == value
            subset_entropy = self.calculate_entropy(y[subset_indices])
            entropy_feature += (np.sum(subset_indices) / total_samples) * subset_entropy

        # Calculate information gain
        information_gain = total_entropy - entropy_feature
        return information_gain

    def find_best_split(self, X, y):
        """
        Find the best feature to split on based on information gain.
        """
        features = X.columns
        max_information_gain = -1
        best_feature = None
        for feature in features:
            information_gain = self.calculate_information_gain(X, y, feature)
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_feature = feature
        return best_feature

    def split_dataset(self, X, y, feature, value):
        """
        Split the dataset based on the chosen feature and its value.
        """
        left_indices = X[feature] == value
        right_indices = ~left_indices
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]
        return X_left, y_left, X_right, y_right

    def build_tree(self, X, y, depth=0):
        """
        Recursively build the Decision Tree.
        """
        # Check for stopping criteria
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return np.bincount(y).argmax()

        # Find the best feature to split on
        best_feature = self.find_best_split(X, y)

        # Split the dataset based on the best feature
        unique_values = np.unique(X[best_feature])
        node = {best_feature: {}}
        for value in unique_values:
            X_left, y_left, X_right, y_right = self.split_dataset(X, y, best_feature, value)
            node[best_feature][value] = self.build_tree(X_left, y_left, depth + 1), self.build_tree(X_right, y_right, depth + 1)

        return node

    def predict(self, tree, sample):
        """
        Make predictions using the trained Decision Tree.
        """
        if not isinstance(tree, dict):
            return tree

        feature = list(tree.keys())[0]
        value = sample[feature]
        if value not in tree[feature]:
            return None

        sub_tree = tree[feature][value]
        return self.predict(sub_tree, sample)

# Example usage:
if __name__ == "__main__":
    # Assuming data is loaded from a CSV file
    data = pd.read_csv("/content/drive/MyDrive/Bangli-P10_gabor.csv", nrows=400)

    # Assuming the second column is the target variable
    y = data.iloc[:, 1]     # Target
    X = data.iloc[:, 2:32]  # Features

    # Create an instance of DecisionTree
    dt = DecisionTree(max_depth=3)

    # Build the Decision Tree
    tree = dt.build_tree(X, y)

    # Make predictions
    sample = X.iloc[0]
    prediction = dt.predict(tree, sample)
    print("Prediction for sample:", prediction)