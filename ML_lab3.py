import pandas as pd
import numpy as np

df = pd.read_excel('C:/Users/Pc/Downloads/Lab Session1 Data.xlsx')

m1 = df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values
m2 = df[['Payment (Rs)']].values
print(m1)
print(m2)
num_rows, num_columns = m1.shape
print("Dimensionality of the vector space:", num_columns)
print("The number of vectors that exist in the vector space:", num_rows)
np_matrix = df.to_numpy()
rank = np.linalg.matrix_rank(m1)
print("Rank of the matrix:", rank)
pI=np.linalg.pinv(m1)
X=pI@m2
print("The individual cost of a candy is: ",round(X[0][0]))
print("The individual cost of a mango is: ",round(X[1][0]))
print("The individual cost of a milk packet is: ",round(X[2][0]))





