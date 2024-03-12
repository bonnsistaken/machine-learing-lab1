# Importing necessary libraries
import pandas as pd
import numpy as np

# Reading data from an Excel file, considering columns B to E
data = pd.read_excel("/content/drive/MyDrive/Lab Session1 Data (1).xlsx", usecols="B:E")

# Displaying the loaded data
print(data)

# Extracting relevant columns from the DataFrame and creating matrices A and C
A = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = data[['Payment (Rs)']].values

# Displaying matrices A and C
print("Matrix A:\n", A)
print("Matrix C:\n", C)

# Determining the dimensionality of the vector space
dimensionality = A.shape[1]
print("Dimensionality of the vector space:", dimensionality)

# Determining the number of vectors in the vector space
num_vectors = A.shape[0]
print("Number of vectors in the vector space:", num_vectors)

# Calculating the rank of matrix A
rank_A = np.linalg.matrix_rank(A)
print("Rank of Matrix A:", rank_A)

# Computing the pseudo-inverse of matrix A
A_pseudo_inv = np.linalg.pinv(A)

# Calculating the cost per product using the pseudo-inverse
cost_per_product = np.dot(A_pseudo_inv, C)
print("Cost of each product available for sale:", cost_per_product)

# Calculating the model vector X for predicting product costs
model_vector_X = np.dot(A_pseudo_inv, C)
print("Model vector X for predicting product costs:", model_vector_X)


# question 3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def classifier(df):
    # Define the features (independent variables) to use in the model
    features = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    X = df[features]  # Extract features data
    y = df['Category']  # Extract target variable (dependent variable)

    # Split the dataset into training and testing sets, allocating 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize the logistic regression classifier
    classifier = LogisticRegression()

    # Train the classifier using the training data
    classifier.fit(X_train, y_train)

    # Make predictions on the entire dataset and store them in a new column
    df['Predicted Category'] = classifier.predict(X)

    return df

# Load the dataset from an Excel file into a pandas DataFrame
df = pd.read_excel("/content/drive/MyDrive/Lab Session1 Data (1).xlsx")

# Create a new 'Category' column in the DataFrame based on the 'Payment (Rs)' column
# Assign 'RICH' if payment is more than 200, otherwise 'POOR'
df['Category'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

# Apply the classifier function to the DataFrame to predict categories
df = classifier(df)

# Print the DataFrame showing the original customer details, payment, actual, and predicted categories
print(df[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Category', 'Predicted Category']]

# Question 4


      
import pandas as pd
import statistics
import matplotlib.pyplot as plt

# Define the path to the Excel file
excel_file_path = "/content/drive/MyDrive/Lab Session1 Data (1).xlsx"

# Read the specified sheet 'IRCTC Stock Price' from the Excel file into a DataFrame
df = pd.read_excel("/content/drive/MyDrive/Lab Session1 Data (1).xlsx", sheet_name='IRCTC Stock Price')

# Calculate and print the mean and variance of the 'Price' column
price_mean = statistics.mean(df['Price'])
price_variance = statistics.variance(df['Price'])
print(f"Mean of Price: {price_mean}\n")
print(f"Variance of Price: {price_variance}\n")

# Extract data for Wednesdays, calculate and print the population mean of 'Price' and the sample mean of 'Price' on Wednesdays
wednesday_data = df[df['Day'] == 'Wed']
wednesday_mean = statistics.mean(wednesday_data['Price'])
print(f"Population Mean of Price: {price_mean}\n")
print(f"Sample Mean of Price on Wednesdays: {wednesday_mean}\n")

# Extract data for April, calculate and print the population mean of 'Price' and the sample mean of 'Price' in April
april_data = df[df['Month'] == 'Apr']
april_mean = statistics.mean(april_data['Price'])
print(f"Population Mean of Price: {price_mean}\n")
print(f"Sample Mean of Price in April: {april_mean}\n")

# Calculate and print the probability of making a loss, the probability of making a profit on Wednesdays, and the conditional probability of making a profit given today is Wednesday
loss_probability = len(df[df['Chg%'] < 0]) / len(df)
print(f"Probability of making a loss: {loss_probability}\n")
wednesday_profit_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(wednesday_data)
print(f"Probability of making a profit on Wednesday: {wednesday_profit_probability}\n")
conditional_profit_probability = wednesday_profit_probability / loss_probability
print(f"Conditional Probability of making profit, given today is Wednesday: {conditional_profit_probability}\n")

# Create a scatter plot of 'Chg%' against the day of the week
day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
day1 = []
chg1 = []

# Iterate through the days and DataFrame rows to extract 'Day' and 'Chg%' values
for i in day:
    for j in range(2, len(df['Day'])):
        if i == df.loc[j, 'Day']:
            day1.append(i)
            chg1.append(df.loc[j, 'Chg%'])
# Plot the scatter plot
plt.scatter(day1, chg1)
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Scatter plot of Chg% against the day of the week')
plt.show()




