import numpy as np

A = np.array([[2,3,1],[4,5,6],[7,8,9]])

determinant = np.linalg.det(A)
inverse = np.linalg.inv(A)

print(f"Determinant: {determinant}")
print(f"Inverse: {inverse}")

B = np.array([[4,-2],[1,1]])

eigvals, eigvec = np.linalg.eig(A)
print(f"Eigenvalues: {eigvals}")
print(f"Eigvectors: {eigvec}")

C = np.array([[10,15,35],[-5,60,6],[12,90,13]])

U, S, Vt = np.linalg.svd(C)
print(f"U:\n {U}")
print(f"Singular Values:\n {S}")
print(f"Transpose:\n {Vt}")

# reconstructed matrix
print(f"Reconsutructed matrix:\n{U @ np.diag(S) @ Vt}")

