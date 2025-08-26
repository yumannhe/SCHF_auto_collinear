import numpy as np
import numpy.linalg as la


'''
generate random basis for 2x2 square lattice.

The logic is the same as in the translational symmetric case.
It's just that the case is more complicated, so we load previously generated basis.

we have 12 basis in total, with 2 in A1, 1 in B1, 1 in B2, 4 in E1 and E2 respectively with every 2 a pair.

The orthonormal spatial basis do not naturally have a uniform basis, 
so we first generate the uniform basis for ferro basis,
and also construct positive, symmetric but non-uniform basis as ferri basis.

On base of the above, we can generate the random basis for 2x2 square lattice.
'''
# load the spatial basis
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
basis = np.load(os.path.join(project_root, 'functions_parameters', '2_2_uc_irrep_basis_spatial.npy'))

# random seed
rng = np.random.default_rng(99)

norb = basis.shape[0]
# construct the uniform basis 
# uniform_basis_coeff = (basis[0,1]+basis[1,1])/(basis[0,1]-basis[1,1])
# uniform_basis = -basis[:,0]-basis[:,1]*uniform_basis_coeff
# ferro_basis = np.stack((uniform_basis, -uniform_basis))
uniform_basis = np.ones((norb,))/np.sqrt(norb)
ferro_basis = np.stack((uniform_basis, -uniform_basis))

# find ferri basis
ferri_basis = np.stack((-basis[:,1]-basis[:,0], basis[:,1]+basis[:,0]))
ferri_basis /= la.norm(ferri_basis, axis=1, keepdims=True)

# construct randomly combined 2d basis
random_coef_2d = rng.uniform(-1, 1, size=(4,2))
random_coef_2d = random_coef_2d/np.linalg.norm(random_coef_2d, axis=1, keepdims=True)
basis_2d = np.stack((basis[:,4:6].T, basis[:,6:8].T, basis[:,8:10].T, basis[:,10:].T))
random_basis_2d = np.einsum('ij,ijk->ik', random_coef_2d, basis_2d)

# obtain the paramagnetic basis:
random_basis_p = np.concatenate((basis[:,2:4].T,random_basis_2d), axis=0)
random_basis_p = np.stack((random_basis_p, random_basis_p), axis=1)

# construct ferri basis for each channel
random_ferri_coef = rng.uniform(0, 1, size=(1,))
random_basis_ferri = ferri_basis[np.newaxis,:,:]*random_ferri_coef + random_basis_p
random_basis_ferri /= la.norm(random_basis_ferri, axis=2, keepdims=True)
random_basis_ferri = np.concatenate((random_basis_ferri, ferri_basis.reshape(1, 2, norb)), axis=0)

# construct ferro basis for each channel
random_ferro_coef = rng.uniform(0, 1, size=(1,))
random_basis_ferro = ferro_basis[np.newaxis,:,:]*random_ferro_coef + random_basis_p
random_basis_ferro /= la.norm(random_basis_ferro, axis=2, keepdims=True)
random_basis_ferro = np.concatenate((random_basis_ferro, ferro_basis.reshape(1, 2, norb)), axis=0)

# put everything together
total_channel_arr = np.concatenate((random_basis_p, random_basis_ferro, random_basis_ferri), axis=0)

np.save('random_basis_arr_2_2.npy', total_channel_arr)
