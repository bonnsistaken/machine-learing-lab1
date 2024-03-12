# question 1

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load your dataset
df = pd.read_csv("/content/drive/MyDrive/Bangli-P10_gabor.csv")

# Choose any two classes for binary classification
class_label1 = 'bad'
class_label2 = 'medium'

# Select rows corresponding to the chosen classes
class_data = df[(df['Original Image'] == class_label1) | (df['Original Image'] == class_label2)]

# Features and class labels for the chosen classes
X_selected = class_data.iloc[:, 2:].values  # Assuming features start from column index 2
y_selected = class_data['Original Image'].map({class_label1: 0, class_label2: 1}).values

# Check if there are enough samples for splitting
if len(X_selected) < 2:
    print("Insufficient samples for splitting. Please check your dataset.")
else:
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.3, random_state=42)

    # Create a kNN classifier with k=3
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier on the training set
    knn_classifier.fit(X_train, y_train)

    # Predictions on training set
    y_train_pred = knn_classifier.predict(X_train)

    # Predictions on test set
    y_test_pred = knn_classifier.predict(X_test)

    # Confusion matrix for training set
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    print("Confusion Matrix (Training Set):")
    print(conf_matrix_train)

    # Confusion matrix for test set
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix (Test Set):")
    print(conf_matrix_test)

    # Precision, Recall, and F1-Score for training set
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    f1_score_train = f1_score(y_train, y_train_pred)

    # Precision, Recall, and F1-Score for test set
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_score_test = f1_score(y_test, y_test_pred)

    # Print performance metrics
    print("\nPerformance Metrics (Training Set):")
    print(f"Precision: {precision_train}")
    print(f"Recall: {recall_train}")
    print(f"F1-Score: {f1_score_train}")

    print("\nPerformance Metrics (Test Set):")
    print(f"Precision: {precision_test}")
    print(f"Recall: {recall_test}")
    print(f"F1-Score: {f1_score_test}")

# question 2

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Assuming actual_prices and predicted_prices are your lists
actual_prices = [100, 150, 200, 120, 180]
predicted_prices = [110, 140, 190, 130, 170]

# Convert lists to numpy arrays for easier calculations
y_actual = np.array(actual_prices)
y_pred = np.array(predicted_prices)

# Calculate MSE (Mean Squared Error)
mse = mean_squared_error(y_actual, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# Calculate R-squared (R2) score
r2 = r2_score(y_actual, y_pred)
print(f"R-squared (R2) Score: {r2}")

# question 3

import numpy as np
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

# Generate 20 data points with random values between 1 and 10 for X and Y
X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20)

# Assign points to classes based on a simple condition (e.g., X + Y > 12)
classes = np.where(X + Y > 12, 1, 0)

# Color mapping for the scatter plot
colors = np.where(classes == 1, 'red', 'blue')

# Scatter plot
plt.scatter(X, Y, c=colors)
plt.title('Scatter Plot of Training Data')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.show()

#question 4

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1
test_X = np.arange(0, 10.1, 0.1)
test_Y = np.arange(0, 10.1, 0.1)

# Create a meshgrid for all possible combinations of X and Y values
test_X, test_Y = np.meshgrid(test_X, test_Y)

# Flatten the meshgrid to create the test set
test_set = np.c_[test_X.ravel(), test_Y.ravel()]

# Create a kNN classifier with k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Use the training data to fit the classifier
knn_classifier.fit(np.column_stack((X, Y)), classes)

# Predict the classes for the test set
predicted_classes = knn_classifier.predict(test_set)

# Color mapping for the scatter plot
predicted_colors = np.where(predicted_classes == 1, 'red', 'blue')

# Scatter plot of the test data with predicted class colors
plt.scatter(test_set[:, 0], test_set[:, 1], c=predicted_colors, alpha=0.1)
plt.title('Scatter Plot of Test Data with Predicted Classes')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.show()

#  question 5 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate training set data with 20 points
np.random.seed(42)
X_train = np.random.uniform(1, 10, 20)
Y_train = np.random.uniform(1, 10, 20)
classes_train = np.random.choice([0, 1], 20)

# Create a kNN classifier with varying values of k
k_values = [1, 3, 5, 7]

plt.figure(figsize=(12, 8))

for k in k_values:
    # Create a kNN classifier with k
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier with training data
    knn_classifier.fit(np.column_stack((X_train, Y_train)), classes_train)

    # Predict the classes for the test set
    predicted_classes = knn_classifier.predict(test_set)

    # Color mapping for the scatter plot
    predicted_colors = np.where(predicted_classes == 1, 'red', 'blue')

    # Scatter plot of the test data with predicted class colors and decision boundaries
    plt.scatter(test_set[:, 0], test_set[:, 1], c=predicted_colors, alpha=0.1, label=f'k={k}')

plt.title('Scatter Plot of Test Data with Predicted Classes and Decision Boundaries')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.legend()
plt.show()

# question 6


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
# Replace 'YourDatasetPath' with the actual path to your dataset
df = pd.read_csv("/content/drive/MyDrive/Bangli-P10_gabor.csv")

# Choose any two classes for binary classification
class_label1 = 'Original Image'
class_label2 = 'Gabor5'

# Select rows corresponding to the chosen classes
class_data = df[(df['Original Image'] == class_label1) | (df['Original Image'] == class_label2)]

# Features and class labels for the chosen classes
X = class_data[['Gabor1', 'Gabor2']].values
y = class_data['Original Image'].values

# Generate training set data with 20 points
np.random.seed(42)
X_train = np.random.uniform(1, 10, 20)
Y_train = np.random.uniform(1, 10, 20)
classes_train = np.random.choice([0, 1], 20)

# Make sure the selected classes are present in the test set
X_test = np.random.uniform(1, 10, 20)  # modify the size if needed
Y_test = np.random.uniform(1, 10, 20)
classes_test = np.random.choice([0, 1], 20)

# Create a kNN classifier with varying values of k
k_values = [1, 3, 5, 7]

plt.figure(figsize=(12, 8))

for k in k_values:
    # Create a kNN classifier with k
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier with training data
    knn_classifier.fit(np.column_stack((X_train, Y_train)), classes_train)

    # Predict the classes for the test set
    predicted_classes = knn_classifier.predict(np.column_stack((X_test, Y_test)))

    # Color mapping for the scatter plot
    predicted_colors = np.where(predicted_classes == 1, 'red', 'blue')

    # Scatter plot of the test data with predicted class colors and decision boundaries
    plt.scatter(X_test, Y_test, c=predicted_colors, alpha=0.1, label=f'k={k}')

plt.title('Scatter Plot of Test Data with Predicted Classes and Decision Boundaries')
plt.xlabel('Feature Gabor1')
plt.ylabel('Feature Gabor2')
plt.legend()
plt.show()

# question 7


import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Generate random data for demonstration
data = {
    'Original Image': np.random.choice(['bad', 'medium'], size=100),
    'Gabor1': np.random.rand(100),
    'Gabor2': np.random.rand(100),
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and class labels
X = df[['Gabor1', 'Gabor2']]
y = df['Original Image']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Set up the kNN classifier
knn_classifier = KNeighborsClassifier()

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_neighbors': np.arange(1, 20),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    knn_classifier, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Display the best parameters
print("Best Parameters:", random_search.best_params_)


# lab 6 

# question 1


import numpy as np

class Perceptron:

  def __init__(self, learning_rate=0.1):
    self.learning_rate = learning_rate
    self.weights = None

  def fit(self, X, y):
    """
    Trains the perceptron model on the given data.

    Args:
      X: A numpy array of shape (n_samples, n_features) representing the training data.
      y: A numpy array of shape (n_samples,) representing the target outputs.
    """
    self.weights = np.random.rand(X.shape[1] + 1)  # Add bias term

    epochs = 0
    while True:
      total_error = 0
      for i in range(len(X)):
        x = X[i]
        target_output = y[i]

        # Calculate weighted sum
        z = np.dot(self.weights[1:], x) + self.weights[0]  # Include bias

        # Apply step activation function
        predicted_output = 1 if z >= 0 else 0

        # Calculate error
        error = target_output - predicted_output

        # Update weights
        self.weights += self.learning_rate * error * np.append(x, 1)  # Include bias update

        total_error += abs(error)

      epochs += 1
      # Stopping criteria: Either low error or maximum epochs reached
      if total_error == 0 or epochs > 100:
        break

  def predict(self, X):
    """
    Predicts the output for the given data points.

    Args:
      X: A numpy array of shape (n_samples, n_features) representing the data points.

    Returns:
      A numpy array of shape (n_samples,) containing the predicted outputs.
    """
    z = np.dot(self.weights[1:], X.T) + self.weights[0]  # Include bias
    return np.where(z >= 0, 1, 0)

# Example usage (assuming your data is preprocessed and normalized)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate target outputs

perceptron = Perceptron()
perceptron.fit(X, y)

predictions = perceptron.predict(X)
print(f"Predicted outputs: {predictions}")

# question 2


import numpy as np

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sigmoid derivative for updates
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Sample inputs (Candies, Mangoes, Milk Packets)
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

# Expected outputs, converted Yes/No to 1/0
y = np.array([[1, 1, 1, 0, 1, 0, 1, 1, 0, 0]]).T

# Initialize weights randomly and bias to zero (for example)
np.random.seed(1)  # For consistent results
weights = np.random.rand(3, 1)
bias = 0
learning_rate = 0.1

# Training loop
for epoch in range(10000):  # Number of iterations
    inputs = X
    weighted_sum = np.dot(inputs, weights) + bias
    outputs = sigmoid(weighted_sum)

    # Calculate the error
    error = y - outputs

    # Adjust weights and bias
    adjustments = error * sigmoid_derivative(outputs)
    weights += np.dot(inputs.T, adjustments) * learning_rate
    bias += np.sum(adjustments) * learning_rate

# Display final weights
print("Weights after training:")
print(weights)
print("\nBias after training:")
print(bias)


# question 3

import numpy as np

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input datasets
inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])
expected_output = np.array([[0],[0],[0],[1]])

epochs = 10000
learning_rate = 0.05
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))

# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * learning_rate
    hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * learning_rate

print("Final hidden weights: ", hidden_weights)
print("Final hidden bias: ", hidden_bias)
print("Final output weights: ", output_weights)
print("Final output bias: ", output_bias)
print("\nOutput from neural network after 10,000 epochs: ",predicted_output)

#  question 4


import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to train and evaluate MLPClassifier
def train_and_evaluate(X_train, y_train, X_test, y_test, problem_description):
    # Initialize MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', max_iter=1000, random_state=42)

    # Train the model
    mlp.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = mlp.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {problem_description}: {accuracy}")

# Load your project dataset
# Replace 'your_dataset.csv' with the actual file path or URL
file_path = '/content/drive/MyDrive/Bangli-P10_gabor.csv'
data_project = pd.read_csv(file_path)

# Extract features and target from the project dataset
X_project = data_project.iloc[:, :-1].values
y_project = data_project.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train_project, X_test_project, y_train_project, y_test_project = train_test_split(
    X_project, y_project, test_size=0.2, random_state=42
)

# A10: MLP for AND gate logic
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])

train_and_evaluate(X_and, y_and, X_and, y_and, "AND Gate Logic with MLP")

# A10: MLP for XOR gate logic
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

train_and_evaluate(X_xor, y_xor, X_xor, y_xor, "XOR Gate Logic with MLP")

# A11: MLP for project dataset
# Assuming the last column in your project dataset is the target variable
X_train, X_test, y_train, y_test = train_test_split(
    X_project, y_project, test_size=0.2, random_state=42
)

train_and_evaluate(X_train, y_train, X_test, y_test, "Project Dataset with MLP")




