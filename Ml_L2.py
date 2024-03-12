import numpy as np
def Manhattendist():   # a function to get the manhatten distance
    F=[]
    S=[]
    print("enter the noof vectors")
    a=int(input())
    for i in range(a):    #a loop to get the vectors
        b=int(input())
        F.append(b)

    H=np.array(F)
    print("enter the second vector")

    for i in range(a):
        c=int(input())
        S.append(c)

    N=np.array(S)     
    Z=H-N            # numpy operation to get the difference of vectors
    print(Z)
    L=len(Z)

    for i in range (L):
        if(Z[i] < 0):
            Z[i]=-1*Z[i]     #code for modulus
    return z
def Euclidian():     # a function to get the eucludian distance

    l1=[]
    l2=[]
    print("enter noof vectors")
    a1=int(input())
    for i in range(a1):
        b1=int(input())
        l1.append(b1)

    L1=np.array(l1)
    print("enter the second vector")
    for i in range(a1):
        c1=int(input())
        l2.append(c1)

    L2=np.array(l2)

    z2=L1-L2
    for i in range(a1):     #operations for calculations
        if (z2[i] ):
            z2[i]=z2[i]*z2[i]
    k=np.sqrt(z2)
    return k
print(Euclidian())
print(Manhattendist())

#question 2
def label_encode(data):
    # Dictionary to store label mappings
    labels = {}
    counter = 0  # Counter for assigning labels
    encoded_data = []  # List to store encoded labels

    for category in data:
        if category not in labels:
            labels[category] = counter  # Assign a new label
            counter += 1
        encoded_data.append(labels[category])

    return encoded_data

if __name__ == "__main__":
    input_data = ["red", "green", "green", "red", "blue","red","yellow"]
    result = label_encode(input_data)

    # Display the result
    print("Original:", input_data)
    print("Encoded:", result)


#Question 3

def one_hot_encode(categories):
    """Convert categories to One-Hot encoded vectors."""
    # Get unique categories and map to indices
    unique_categories = sorted(set(categories))
    category_to_index = {category: index for index, category in enumerate(unique_categories)}

    # Initialize list for One-Hot encoded vectors
    one_hot_encoded = []

    # Encode each category
    for category in categories:
        encoded_vector = [0] * len(unique_categories)  # Vector of zeros
        encoded_vector[category_to_index[category]] = 1  # Set index of category to 1
        one_hot_encoded.append(encoded_vector)

    return one_hot_encoded

# Main program
if __name__ == "__main__":
    categories = ["dog", "cat", "bird", "dog"]  # Example categories
    encoded_vectors = one_hot_encode(categories)  # Encode categories
    print("One-Hot Encoded Vectors:")  # Output results
    for vector in encoded_vectors:
        print(vector)

# Question 4


def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    distance = 0
    for i in range(len(point1) - 1):  # Exclude the label from the distance calculation
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

def get_neighbors(training_data, test_point, k):
    """Find the k nearest neighbors of a test point."""
    distances = []
    for train_point in training_data:
        distance = calculate_distance(test_point, train_point)
        distances.append((train_point, distance))
    distances.sort(key=lambda tup: tup[1])  # Sort by distance
    neighbors = distances[:k]
    return [neighbor[0] for neighbor in neighbors]  # Return only the data points, not distances

def predict_classification(training_data, test_point, k):
    """Predict the classification for a test point based on k nearest neighbors."""
    neighbors = get_neighbors(training_data, test_point, k)
    votes = {}  # To count the votes for each class
    for neighbor in neighbors:
        label = neighbor[-1]  # The class label is the last element in the tuple
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
    sorted_votes = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_votes[0][0]  # Return the label with the most votes

# Example usage
if __name__ == "__main__":
    # Example training dataset [(feature1, feature2, ..., label), ...]
    training_data = [(1, 2, 'A'), (2, 3, 'A'), (3, 4, 'B'), (4, 5, 'B')]
    # Test point to classify
    test_point = (3, 3)
    k = 3  # Number of neighbors to consider
    predicted_label = predict_classification(training_data, test_point + (None,), k)
    print(f"Predicted Class for the test point {test_point}: {predicted_label}")

