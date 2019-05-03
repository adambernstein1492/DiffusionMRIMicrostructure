import numpy as np
import nibabel as nib
from . import spherical_harmonics as SH
from . import util
import os

def main_qbi(dwi_file, bval_file, bvec_file, mask_file, out_path,
             order=8, calc_GFA=True, save_glyphs=True):

    # Load in data
    dwi, mask, bvals, bvecs = util.load_diffusion_data(dwi_file, bval_file, bvec_file, mask_file)
    data = dwi.get_data()
    mask = mask.get_data()

    # Select Shell with Highest b-value
    data, bvals, bvecs = util.select_largest_shell(data, bvals, bvecs)

    # Calculate FRT Matrix 'P'
    P = calc_P_matrix(order)

    # Fit the Signal using SH
    sig_coeffs = SH.fit_to_SH(data, bvecs, mask, order)

    # Multiply Fit Signal by P to get FRT
    QB_coeffs = FRT(sig_coeffs, P, mask)

    # Min Max Normalize
    QB_coeffs = min_max_normalize(QB_coeffs, mask, order)

    # Save Outputs
    if save_glyphs:
        img = nib.Nifti1Image(QB_coeffs, dwi.affine, dwi.header)
        nib.save(img, out_path + 'QBI_Glyphs.nii')

    if calc_GFA:
        print("Calculating GFA")
        gfa = calc_gfa(QB_coeffs, mask, order)

        img = nib.Nifti1Image(gfa, dwi.affine, dwi.header)
        nib.save(img, out_path + 'QBI_GFA.nii')

def calc_P_matrix(order):
    num_harmonics = (order + 1) * (order + 2) / 2
    P = np.zeros((num_harmonics, num_harmonics))

    # Calculate Legendre Polynomial of degree L at 0 'P_l(0)'
    index = 0
    for L in range(0,order+1,2):
        P_l = 2 * np.pi * (-1.0) ** (L/2.0) * util.factn(L-1,2) / util.factn(L,2)

        for m in range(-L,L+1):
            P[index,index] = P_l
            index += 1

    return P

def FRT(sig_coeffs, P, mask):

    # Used for Progress update
    count = 0.0
    percent_prev = 0.0
    num_vox = np.sum(mask)

    QB_coeffs = np.zeros(sig_coeffs.shape)
    for x in range(sig_coeffs.shape[0]):
        for y in range(sig_coeffs.shape[1]):
            for z in range(sig_coeffs.shape[2]):
                if mask[x,y,z] != 0:
                    QB_coeffs[x,y,z,:] = np.matmul(P, sig_coeffs[x,y,z,:])

                    # Update Progress
                    count += 1.0
                    percent = np.around((count / num_vox * 100), decimals=1)
                    if(percent != percent_prev):
                        util.progress_update("Performing FRT: ", percent)
                        percent_prev = percent

    return QB_coeffs

def min_max_normalize(coeffs, mask, order):
    # Sample ODF at given points
    file_location = os.path.dirname(__file__)
    sample_dirs = np.array(util.read_direction_file(file_location + "/../direction_files_qsi/642vertices.txt"))

    odf = SH.eval_SH_basis(coeffs, sample_dirs, mask, order)

    for x in range(coeffs.shape[0]):
        for y in range(coeffs.shape[1]):
            for z in range(coeffs.shape[2]):
                if mask[x,y,z] != 0:
                    odf[x,y,z,:] -= np.amin(odf[x,y,z,:])
                    odf[x,y,z,:] /= np.amax(odf[x,y,z,:])

    # Refit ODF using SH
    QB_coeffs = SH.fit_to_SH(odf, sample_dirs, mask, order)

    return QB_coeffs

def calc_gfa(QB_coeffs, mask, order):
    # Sample ODF at given points
    file_location = os.path.dirname(__file__)
    sample_dirs = np.array(util.read_direction_file(file_location + "/../direction_files_qsi/642vertices.txt"))

    ODF = SH.eval_SH_basis(QB_coeffs, sample_dirs, mask, order)

    mean_val = np.mean(ODF,axis=3)

    mss = np.zeros((ODF.shape[0], ODF.shape[1], ODF.shape[2]))
    ss = np.zeros((ODF.shape[0], ODF.shape[1], ODF.shape[2]))
    for i in range(ODF.shape[3]):
        mss += ODF.shape[3] * (ODF[:,:,:,i] - mean_val) ** 2
        ss += (ODF.shape[3] - 1) * ODF[:,:,:,i] ** 2

    gfa = np.divide(mss, ss, out = np.zeros_like(mss), where=ss!=0)
    gfa[np.isnan(gfa)] = 0

    return gfa
