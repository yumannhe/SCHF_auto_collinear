import numpy as np
import numpy.linalg as la

# lattice vector
# convention: each row is a lattice vector in Cartesian coordinate systems
a = np.array([[5.279, 0],
              [-2.6395, 4.571748]])
b = (la.inv(a) * 2 * np.pi).T

# including Pauli matrices
sigma_0 = np.array([[1.0, 0], [0, 1.0]], np.complex128)
sigma_1 = np.array([[0, 1.0], [1.0, 0]], np.complex128)
sigma_2 = np.array([[0, -1.0 * 1j], [1.0 * 1j, 0]], np.complex128)
sigma_3 = np.array([[1.0, 0], [0, -1.0]], np.complex128)
pauli_matrices = np.stack([sigma_0, sigma_1, sigma_2, sigma_3])

# document basis in 1*1 unit cell
a1_basis = np.ones(3)/np.sqrt(3)
e2_basis1 = np.array([-1, 2, -1]) / np.sqrt(6)
e2_basis2 = np.array([-1, 0, 1]) / np.sqrt(2)

