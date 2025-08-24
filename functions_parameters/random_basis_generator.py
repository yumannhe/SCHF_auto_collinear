import numpy as np
import numpy.linalg as la
from functions_parameters.universal_parameters import a1_basis, e2_basis1, e2_basis2

'''
In the translational-symmetric case, we have 3 orthonormal basis:
1 uniform a1_basis and 2 e2_basis1, e2_basis2.

We try to generate random basis based on the following rules:
1. The random basis should be norm 1.
2. We hope to distinguish between ferro and ferri channels.
3. We neglect completely symmetric channels.

Therefore, naturally we have only 1 ferro channel obtained from a1_basis.

Then for the e2 basis, we allow random mixing between e2_basis1 and e2_basis2.
and then we add uniform ferro basis to form the ferro channel.
To ensure Ferri without AFM, we generate random positive numbers for the ferri channel.
'''

# set up the seed
rng = np.random.default_rng(111)      
# allowed number of mixing for e2 basis
num_mixing = 3
# norb
norb = a1_basis.shape[0]
# e2_basis
e2_basis = np.stack((e2_basis1, e2_basis2), axis=0)
# ferro basis
ferro_basis = np.stack((a1_basis, -a1_basis), axis=0)

# generate random coefficients for e2 basis
random_coefficients_e2 = rng.uniform(-1, 1, size=(num_mixing, 2))
random_coefficients_e2 /= la.norm(random_coefficients_e2, axis=1, keepdims=True)
# generate time reversal even (paramagnetic) random e2 basis
basis_e2_p = random_coefficients_e2@e2_basis
basis_e2_p = np.stack((basis_e2_p, basis_e2_p), axis=1)

# generate random coefficients for ferro basis
random_coefficients_ferro = rng.uniform(0, 1, size=(1,))
# generate nematic ferro basis
basis_e2_ferro = random_coefficients_ferro*ferro_basis + basis_e2_p
basis_e2_ferro /= la.norm(basis_e2_ferro, axis=2, keepdims=True)

# generate random coefficients for ferri basis
basis_e2_ferri = rng.uniform(0, 1, size=(num_mixing, norb))
basis_e2_ferri /= la.norm(basis_e2_ferri, axis=1, keepdims=True)
basis_e2_ferri = np.stack((basis_e2_ferri, -basis_e2_ferri), axis=1)

total_channel_arr = np.concatenate((ferro_basis[np.newaxis,:,:], basis_e2_p, basis_e2_ferro, basis_e2_ferri), axis=0)

np.save('random_basis_arr.npy', total_channel_arr)