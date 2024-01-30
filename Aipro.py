#first problem
l=[2,7,4,1,3,6]
L=len(l)
sum=0
for i in range(L-1):
    for j in range(L-1-i):
        if(l[j]+l[j+1] == 10):  #it just checks for the ==10 and adds to the sum
            sum=sum+1
print(sum)

#second problem

print("enter the noof elements")
l=[]
n=int(input())
if (n<3):  #the program checks for the minimum input 
    print("give more numbers")
else:#else it goes to the code block
    for i in range(n):
      m=int(input())
      l.append(m)
    L=len(l)
    for i in range(L-1):
      for j in range(L-1-i): #what is done is that i sorted the list thst i have enterd and took zeroth index and last
         if (l[j]< l[j+1]):
          temp=l[j]
          l[j]=l[j+1]
          l[j+1]=temp
      print("the sorted list is")
      print(l)
      B=l[0]#
      S=l[L-1]

      diff=(B-S)
      print(diff)#we can get the difference just by taking the last index and first index and difference

#third problem
      
      import numpy as np

def matrix_power(A, m):
    if A.shape[0] != A.shape[1]:#
        return "Error: A must be a square matrix"

    result = np.eye(A.shape[0])

    
    for _ in range(m):
        result = np.dot(result, A)

    return result

A = np.array([[1, 2],
              [3, 4]])#default matrix for multiplying

m = int(input("Enter the exponent (m): ")) #the multiplier

result = matrix_power(A, m)
print(f"A raised to the power of {m} is:")
print(result)

#fourth problem

def count_highest_occurrence(input_string):
    
    count_dict = {}#store it the dict

    
    for char in input_string:
        if char.isalpha():
            if char in count_dict:
                count_dict[char] += 1
            else:
                count_dict[char] = 1

    
    max_count = 0
    max_char = None
    for char, count in count_dict.items():
        if count > max_count:
            max_count = count
            max_char = char

    return max_char, max_count


input_string = input("Enter a string: ") #takes the input string and checks for the noof times that it has been repeated
highest_char, occurrence_count = count_highest_occurrence(input_string.lower())
if highest_char:
    print(f"The highest occurring character is '{highest_char}' with {occurrence_count} occurrences.")
else:
    print("No alphabets found in the input string.")

