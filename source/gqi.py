import numpy as np
import nibabel as nib
from . import spherical_harmonics as SH
from . import util
import os

def main_gqi(dwi_file, bval_file, bvec_file, mask_file, out_path,
             order=8, length_scale=1.25, calc_QA=True, calc_GFA=True,
             save_glyphs=True):

    # Load in data
    dwi, mask, bvals, bvecs = util.load_diffusion_data(dwi_file, bval_file, bvec_file, mask_file)
    data = dwi.get_data()
    mask = mask.get_data()

    # Find Directions to calculate SDF at
    file_location = os.path.dirname(__file__)
    sample_dirs = np.array(util.read_direction_file(file_location + "/../direction_files_qsi/642vertices.txt"))

    # Calculate SDF
    sdfs = calc_sdf(data, bvals, bvecs, mask, sample_dirs)

    #sdfs_img = nib.Nifti1Image(sdfs, dwi.affine, dwi.header)
    #nib.save(sdfs_img, out_path + 'SDFS.nii')

    # Fit SDF using Spherical Harmonics
    if save_glyphs:
        coeffs = SH.fit_to_SH(sdfs, sample_dirs, mask, order)

        img = nib.Nifti1Image(coeffs, dwi.affine, dwi.header)
        nib.save(img, out_path + 'GQI_Glyphs.nii')

    # Calulate and Save Quantitative Anisotropy Image and GFA
    if calc_QA:
        print("Calculating QA")
        qa = calc_qa(sdfs)

        img = nib.Nifti1Image(qa, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'QA.nii'))

    if calc_GFA:
        print("Calculating GFA")
        gfa = calc_gfa(sdfs)

        gfa_img = nib.Nifti1Image(gfa, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'GFA_GQI.nii'))


def calc_sdf(dwi, bvals, bvecs, mask, sample_dirs):

    sinc_proj = proj_sdf_bvec(sample_dirs, bvals, bvecs)

    # Used for Progress update
    count = 0.0
    percent_prev = 0.0
    num_vox = np.sum(mask)

    # Calculate ODF values at Sampling Directions
    sdfs = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], sinc_proj.shape[0]))
    for x in range(dwi.shape[0]):
        for y in range(dwi.shape[1]):
            for z in range(dwi.shape[2]):
                if mask[x,y,z] != 0:
                    sdfs[x,y,z,:] = np.matmul(sinc_proj,dwi[x,y,z,:])

                    # Update Progress
                    count += 1.0
                    percent = np.around((count / num_vox * 100), decimals = 1)
                    if(percent != percent_prev):
                        util.progress_update("Calculating SDFs: ", percent)
                        percent_prev = percent

    for x in range(sdfs.shape[0]):
        for y in range(sdfs.shape[1]):
            for z in range(sdfs.shape[2]):
                if mask[x,y,z] != 0:
                    sdfs[x,y,z,:] -= np.amin(sdfs[x,y,z,:])

    sdfs /= np.amax(sdfs)

    return sdfs


def proj_sdf_bvec(sample_dirs, bvals, bvecs):
    length_scale = np.sqrt(bvals * 0.01506) #0.015 is ~6D for free water at 37 C

    # Scale b-vectors by length scale and diffusion weighting
    b_vector = np.zeros(bvecs.shape)
    for i in range(len(length_scale)):
        b_vector[i] = bvecs[i] * length_scale[i]

    # Inner product of ODF directions and bvector directions
    proj = np.matmul(sample_dirs, np.transpose(b_vector))

    # Sinc values of inner products
    sinc_proj = np.sinc(proj * 1.25 / np.pi)

    return sinc_proj

def calc_qa(sdfs):

    qfa = np.amax(sdfs,axis=3)

    return qfa

def calc_gfa(sdfs):
    mean_val = np.mean(sdfs,axis=3)

    mss = np.zeros((sdfs.shape[0], sdfs.shape[1], sdfs.shape[2]))
    ss = np.zeros((sdfs.shape[0], sdfs.shape[1], sdfs.shape[2]))
    for i in range(sdfs.shape[3]):
        mss += sdfs.shape[3] * (sdfs[:,:,:,i] - mean_val) ** 2
        ss += (sdfs.shape[3] - 1) * sdfs[:,:,:,i] ** 2

    gfa = np.divide(mss, ss, out = np.zeros_like(mss), where=ss!=0)
    gfa[np.isnan(gfa)] = 0

    return gfa
