import numpy as np
from functools import reduce
import numpy.linalg as la

def num2str(x_in, precision="{0:.3E}", num_zero=1E-12, simple_zero=True):
    """
    Convert numbers to str for printing. We use a bigger num_zero since this is only display-related

    :param x_in: int/ float/ complex to be converted
    :param precision: precision for the string
    :param num_zero: treating |x_in| < num_zer as 0
    :param simple_zero: Return integer 0 instead of float
    :return: str for the input
    """
    x = x_in
    if isinstance(x, complex):
        if np.abs(x.imag) < num_zero:
            ret = precision.format(x.real)
        else:
            if x.imag >= 0:
                ret = precision.format(x.real) + '+' + precision.format(x.imag) + ' i'
            else:
                ret = precision.format(x.real) + precision.format(x.imag) + ' i'
    elif isinstance(x, (int, np.integer)):
        ret = str(x)
    else:
        ret = precision.format(x)

    if simple_zero == 1 and np.abs(x) < num_zero:
        ret = '0'

    return ret

def array_print(mat, dim_max=40, fix_width=True):
    """
    Take a numpy array mat and print the rows out with entries converted to str using num2str. The current version is
    more terminal friendly than the previous incarnation based on IPython

    :param mat: np.ndarrau
    :param dim_max: max allowed matrix dimension without prompting user confirmation
    :param fix_width: bool on whether the printed entries all have the same width
    """
    dim_left, dim_right = mat.shape
    if dim_left > dim_max or dim_right > dim_max:
        proceed = input('Matrix dimension will be %d x %d, proceed? [Y/N]: ' % (dim_left, dim_right))
    else:
        proceed = 'Y'

    if proceed == 'Y':
        mat_str = [num2str(x) for x in np.ravel(mat)]
        if fix_width:
            str_width = max([len(s) for s in mat_str])
            mat_str = [s.center(str_width) for s in mat_str]
        mat_str = np.array(mat_str).reshape(mat.shape)

        print('Array = [')
        for mat_str_row in mat_str:
            row_now = reduce(lambda x, y: x + y, [' ' + x + ', ' for x in mat_str_row])
            print(row_now)

        print(']')
    else:
        print('Array too big')
    return None


def fermi_dirac_electron_count(energy_sequence, chemical_potential, t, overflow_threshold=40):
    energy_weight = (energy_sequence - chemical_potential) / t
    energy_weight = np.clip(energy_weight, -overflow_threshold, overflow_threshold)
    density = 1 / (np.exp(energy_weight) + 1)
    return density


def fermi_level_bisection(energy_sequence, target_filling, t, tolerance=1E-9):
    upper_bound = energy_sequence[-1]
    lower_bound = energy_sequence[0]
    # print('Lower bound: %.3E, upper bound: %.3E' % (lower_bound, upper_bound), flush=True)
    fermi_e = 0
    diff = 1
    iteration = 0
    while np.abs(diff) > tolerance and iteration < 100:
        fermi_e = (upper_bound + lower_bound) / 2
        diff = np.sum(fermi_dirac_electron_count(energy_sequence, fermi_e, t))/ energy_sequence.shape[0] - target_filling
        # print('iteration %d: lower bound: %.3E, upper bound: %.3E, fermi_e: %.3E, diff: %.3E' % (iteration, lower_bound, upper_bound, fermi_e, diff), flush=True)
        if diff < 0:
            lower_bound = fermi_e
        else:
            upper_bound = fermi_e
        iteration = iteration + 1
    if iteration == 100:
        print('accuracy not reached! The difference is %.3E' % diff, flush=True)
        print('the minimum energy degeneracy is')
        print(energy_sequence[:50])
    # if np.abs(diff) > tolerance:
    #     print('accuracy not reached! The difference is %.3E' % diff)
    # print(iteration)
    # print(diff)
    return fermi_e


def rot_symm_m_check_d(d, c6, c3, c2):
    '''
    check the rotation symmetry of the density of correlation matrix o 
    c6, c3, c2 are the rotation matrices for the 6-fold, 3-fold, and 2-fold symmetry.
    return the difference between the correlation matrix o and the rotation matrices and magnetic order.
    '''
    density_arr = d[0] + d[1]
    magnetism_arr = d[0] - d[1]
    c6_diff = np.max(np.abs(c6@density_arr - density_arr))
    c3_diff = np.max(np.abs(c3@density_arr - density_arr))
    c2_diff = np.max(np.abs(c2@density_arr - density_arr))
    return c6_diff, c3_diff, c2_diff, magnetism_arr

def rot_symm_m_check_corr_o_diag_bond(corr_o, c6, c3, c2):
    '''
    check the rotation symmetry of the density of correlation matrix o 
    c6, c3, c2 are the rotation matrices for the 6-fold, 3-fold, and 2-fold symmetry.
    return the difference between the correlation matrix o and the rotation matrices and magnetic order.
    '''
    diag_entries = np.diagonal(corr_o, axis1=1, axis2=2)
    density_arr = diag_entries[0] + diag_entries[1]
    magnetism_arr = diag_entries[0] - diag_entries[1]
    c6_diff_diag = np.max(np.abs(c6@density_arr - density_arr))
    c3_diff_diag = np.max(np.abs(c3@density_arr - density_arr))
    c2_diff_diag = np.max(np.abs(c2@density_arr - density_arr))
    bond_diff_spin = np.max(np.abs(corr_o[0] - corr_o[1] - magnetism_arr))
    return c6_diff_diag, c3_diff_diag, c2_diff_diag, magnetism_arr, bond_diff_spin 


def translation_check_d(d, translation_a1, translation_a2):
    '''
    check the translation symmetry of the density of correlation matrix o 
    translation_a1, translation_a2 are the translation matrices for the a1 and a2 basis.
    return the difference between the correlation matrix o and the translation matrices.
    '''
    density_arr = d[0] + d[1]
    magnetism_arr = d[0] - d[1]
    translation_a1_diff_c = np.max(np.abs(translation_a1@density_arr - density_arr))
    translation_a2_diff_c = np.max(np.abs(translation_a2@density_arr - density_arr))
    translation_a1_diff_m = np.max(np.abs(translation_a1@magnetism_arr - magnetism_arr))
    translation_a2_diff_m = np.max(np.abs(translation_a2@magnetism_arr - magnetism_arr))
    return translation_a1_diff_c, translation_a2_diff_c, translation_a1_diff_m, translation_a2_diff_m

