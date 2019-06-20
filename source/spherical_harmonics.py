import numpy as np
import scipy.special
from . import util

def eval_spherical_harmonics(directions, order):
    # Convert to spherical coordinates
    dirs_sphere = cart_to_sphere(directions)
    num_harmonics = int((order + 1) * (order + 2) / 2)

    B = np.zeros((int(directions.shape[0]), num_harmonics))

    index = 0
    for L in range(0,order+1,2):
        for m in range(-L,L+1):
            M = np.absolute(m)
            if m < 0:
                B[:,index] = np.imag(scipy.special.sph_harm(M, L, dirs_sphere[:,1], dirs_sphere[:,0]))
                index += 1
            elif m == 0:
                B[:,index] = np.real(scipy.special.sph_harm(M, L, dirs_sphere[:,1], dirs_sphere[:,0]))
                index += 1
            else:
                B[:,index] = np.real(scipy.special.sph_harm(M, L, dirs_sphere[:,1], dirs_sphere[:,0]))
                index += 1

    return B

def fit_to_SH(signal, directions, mask, order, reg=0.006):

    L = calc_normalization_matrix(order)
    B = eval_spherical_harmonics(directions, order)

    # Used for Progress update
    count = 0.0
    percent_prev = 0.0
    num_vox = np.sum(mask)

    # Fit Coeffs for all signal values
    coeffs = np.zeros((signal.shape[0], signal.shape[1], signal.shape[2], B.shape[1]))
    for x in range(signal.shape[0]):
        for y in range(signal.shape[1]):
            for z in range(signal.shape[2]):
                 if mask[x,y,z] != 0:
                     first_term = np.matmul(B.T, B) + reg * L
                     second_term = np.matmul(B.T, signal[x,y,z,:])

                     coeffs[x,y,z,:] = np.matmul(np.linalg.inv(first_term), second_term)

                     # Update Progress
                     count += 1.0
                     percent = np.around((count / num_vox * 100), decimals = 1)
                     if(percent != percent_prev):
                         util.progress_update("Fitting Spherical Harmonics: ", percent)
                         percent_prev = percent

    return coeffs

def fit_to_SH_MAP(signal, directions, order, reg=0.006):
    # Create Normalization Matrix
    L = calc_normalization_matrix(order)

    # Evaluate Spherical Harmonics at each measured point
    B = eval_spherical_harmonics(directions, order)

    # Fit Coeffs
    first_term = np.matmul(B.T, B) + reg * L
    second_term = np.matmul(B.T, signal)

    coeffs = np.matmul(np.linalg.inv(first_term), second_term)

    return coeffs


def eval_SH_basis(coeffs, directions, mask, order):
    B = eval_spherical_harmonics(directions,order)
    num_harmonics = (order + 1) * (order + 2) / 2

    # Used for Progress update
    count = 0.0
    percent_prev = 0.0
    num_vox = np.sum(mask)
    values = np.zeros((coeffs.shape[0], coeffs.shape[1], coeffs.shape[2], directions.shape[0]))
    for x in range(coeffs.shape[0]):
        for y in range(coeffs.shape[1]):
            for z in range(coeffs.shape[2]):
                if mask[x,y,z] != 0:
                    values[x,y,z,:] = np.matmul(B, coeffs[x,y,z,:])

                    # Update Progress
                    count += 1.0
                    percent = np.around((count / num_vox * 100), decimals = 1)
                    if(percent != percent_prev):
                        util.progress_update("Evaluating SH at Sample Points: ", percent)
                        percent_prev = percent

    return values

def calc_normalization_matrix(order):

    num_harmonics = int((order + 1) * (order + 2) / 2)

    L = np.zeros((num_harmonics, num_harmonics))

    i = 0
    for l in range(0,order+1,2):
        for m in range(-l,l+1):
            L[i,i] = l ** 2 * (l + 1) ** 2
            i += 1

    return L

def cart_to_sphere(directions):
    # Calc Theta
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = np.arccos(directions[:,2])

    # Calc Phi
    phi = np.zeros((directions.shape[0]))

    for i in range(directions.shape[0]):
        ratio = np.absolute(np.divide(directions[i,1], directions[i,0], out=np.zeros_like(directions[i,1]), where=directions[i,0]!=0))
        if ((directions[i,0] >= 0) and (directions[i,1] >= 0)): # 1st Quadrant
            phi[i] = np.arctan(ratio)

        if ((directions[i,0] < 0) and (directions[i,1] >= 0)):  # 2nd Quadrant
            phi[i] = np.pi - np.arctan(ratio)

        if ((directions[i,0] < 0) and (directions[i,1] < 0)):   # 3rd Quadrant
            phi[i] = np.pi + np.arctan(ratio)

        if ((directions[i,0] >= 0) and (directions[i,1] < 0)):  # 4th Quadrant
            phi[i] = (2.0 * np.pi) - np.arctan(ratio)

    sphere_coords = np.zeros((directions.shape[0], 2))
    sphere_coords[:,0] = theta
    sphere_coords[:,1] = phi

    return sphere_coords
