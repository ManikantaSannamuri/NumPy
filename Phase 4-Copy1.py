#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Create 5 matrices with five different dimensions (1-D,2-D,...5-D)
import numpy as np

# 1-D Matrix
matrix_1d = np.array([1, 2, 3, 4, 5])

# 2-D Matrix
matrix_2d = np.array([[1, 2, 3],
                      [4, 5, 6]])

# 3-D Matrix
matrix_3d = np.array([[[1, 2],
                       [3, 4]],
                      
                      [[5, 6],
                       [7, 8]]])

# 4-D Matrix
matrix_4d = np.array([[[[1, 2],
                        [3, 4]],
                       
                       [[5, 6],
                        [7, 8]]]])

# 5-D Matrix
matrix_5d = np.array([[[[1, 2],
                         [3, 4]],
                        
                        [[5, 6],
                         [7, 8]]]])

# Printing the matrices
print("1-D Matrix:")
print(matrix_1d)

print("\n2-D Matrix:")
print(matrix_2d)

print("\n3-D Matrix:")
print(matrix_3d)

print("\n4-D Matrix:")
print(matrix_4d)

print("\n5-D Matrix:")
print(matrix_5d)


# In[5]:


#Find determinants of 5 matrices and display your output
import numpy as np

# Define 5 different matrices
matrix_1 = np.array([[1, 2],
                     [3, 4]])

matrix_2 = np.array([[5, 6, 7],
                     [8, 9, 10],
                     [11, 12, 13]])

matrix_3 = np.array([[2, 0, 0, 0],
                     [0, 3, 0, 0],
                     [0, 0, 4, 0],
                     [0, 0, 0, 5]])

matrix_4 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

matrix_5 = np.array([[1, 0, 0, 0, 0],
                     [0, 2, 0, 0, 0],
                     [0, 0, 3, 0, 0],
                     [0, 0, 0, 4, 0],
                     [0, 0, 0, 0, 5]])

# Calculate the determinants
determinant_1 = np.linalg.det(matrix_1)
determinant_2 = np.linalg.det(matrix_2)
determinant_3 = np.linalg.det(matrix_3)
determinant_4 = np.linalg.det(matrix_4)
determinant_5 = np.linalg.det(matrix_5)

# Display the determinants
print("Determinant of Matrix 1:")
print(determinant_1)

print("\nDeterminant of Matrix 2:")
print(determinant_2)

print("\nDeterminant of Matrix 3:")
print(determinant_3)

print("\nDeterminant of Matrix 4:")
print(determinant_4)

print("\nDeterminant of Matrix 5:")
print(determinant_5)


# In[3]:


import numpy as np

# Define 5 matrices as NumPy arrays
matrices = []

matrix1 = np.array([[1, 2],
                    [3, 4]])

matrix2 = np.array([[5, 6],
                    [7, 8]])

matrix3 = np.array([[9, 10],
                    [11, 12]])

matrix4 = np.array([[13, 14],
                    [15, 16]])

matrix5 = np.array([[17, 18],
                    [19, 20]])

matrices.extend([matrix1, matrix2, matrix3, matrix4, matrix5])

# Calculate and display the inverses of the matrices
for i, matrix in enumerate(matrices, start=1):
    try:
        inverse_matrix = np.linalg.inv(matrix)
        print(f"Inverse of matrix {i}:\n{inverse_matrix}\n")
    except np.linalg.LinAlgError:
        print(f"Matrix {i} is singular and does not have an inverse.\n")


# In[7]:


#find the rank, diagonal elements, and trace of five matrices using NumPy:
import numpy as np

# Define 5 different matrices
matrix_1 = np.array([[1, 2],
                     [3, 4]])

matrix_2 = np.array([[5, 6, 7],
                     [8, 9, 10],
                     [11, 12, 13]])

matrix_3 = np.array([[2, 0, 0, 0],
                     [0, 3, 0, 0],
                     [0, 0, 4, 0],
                     [0, 0, 0, 5]])

matrix_4 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

matrix_5 = np.array([[1, 0, 0, 0, 0],
                     [0, 2, 0, 0, 0],
                     [0, 0, 3, 0, 0],
                     [0, 0, 0, 4, 0],
                     [0, 0, 0, 0, 5]])

# Find the rank, diagonal, and trace for each matrix
for i, matrix in enumerate([matrix_1, matrix_2, matrix_3, matrix_4, matrix_5], start=1):
    print(f"Matrix {i}:")
    print("-------------")
    
    # Rank
    rank = np.linalg.matrix_rank(matrix)
    print(f"Rank: {rank}")
    
    # Diagonal elements
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        diagonal = np.diagonal(matrix)
        print(f"Diagonal: {diagonal}")
    else:
        print("Matrix is not square, so it doesn't have diagonal elements.")
    
    # Trace
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        trace = np.trace(matrix)
        print(f"Trace: {trace}")
    else:
        print("Matrix is not square, so it doesn't have a trace.")
    
    print("\n")


# In[2]:


# Find Eigen value and eigen vector for 5 matrices
import numpy as np

# Define 5 different matrices
matrix_1 = np.array([[1, 2],
                     [3, 4]])

matrix_2 = np.array([[5, 6, 7],
                     [8, 9, 10],
                     [11, 12, 13]])

matrix_3 = np.array([[2, 0, 0, 0],
                     [0, 3, 0, 0],
                     [0, 0, 4, 0],
                     [0, 0, 0, 5]])

matrix_4 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

matrix_5 = np.array([[1, 0, 0, 0, 0],
                     [0, 2, 0, 0, 0],
                     [0, 0, 3, 0, 0],
                     [0, 0, 0, 4, 0],
                     [0, 0, 0, 0, 5]])

# Find eigenvalues and eigenvectors for each matrix
for i, matrix in enumerate([matrix_1, matrix_2, matrix_3, matrix_4, matrix_5], start=1):
    print(f"Matrix {i}:")
    print("-------------")
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    print("Eigenvalues:")
    print(eigenvalues)
    
    print("Eigenvectors:")
    print(eigenvectors)
    
    print("\n")


# In[ ]:




