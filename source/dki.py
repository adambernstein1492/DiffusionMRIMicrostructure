import numpy as np
import nibabel as nib
from . import util
from . import dti

def main_dki(dwi_file, bval_file, bvec_file, mask_file, out_path, b_thresh=2100,
             calc_DTI_params=True, calc_DKI_params=True):

    # Load in data
    dwi, mask, bvals, bvecs = util.load_diffusion_data(dwi_file, bval_file, bvec_file, mask_file)
    data = dwi.get_data()
    mask = mask.get_data()

    # Filter b-values
    data, bvals, bvecs = filter_bvals(data, bvals, bvecs, b_thresh)

    # Fit the data
    dki_tensor, eigen_values, eigen_vectors = fit_dki(data, bvals, bvecs, mask)

    #dki_img = nib.Nifti1Image(dki_tensor, dwi.affine, dwi.header)
    #nib.save(dki_img, (out_path + 'DKI.nii'))

    # Calculate and Save Scalar Maps
    if calc_DTI_params:
        FA = dti.calc_fa(eigen_values)
        MD, AD, RD = dti.calc_diffusivities(eigen_values)

        # Save
        img = nib.Nifti1Image(FA, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'FA_DKI.nii'))

        img = nib.Nifti1Image(MD, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'MD_DKI.nii'))

        img = nib.Nifti1Image(AD, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'AD_DKI.nii'))

        img = nib.Nifti1Image(RD, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RD_DKI.nii'))

    if calc_DKI_params:
        FAK, MK, AK, RK = calc_dki_parameters(dki_tensor)

        img = nib.Nifti1Image(FAK, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'KA.nii'))

        img = nib.Nifti1Image(MK, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'MK.nii'))

        img = nib.Nifti1Image(AK, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'AK.nii'))

        img = nib.Nifti1Image(RK, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RK.nii'))

    return dki_tensor, eigen_values, eigen_vectors

def fit_dki(dwi, bvals, bvecs, mask):
    # Calculate the b-matrix
    b_matrix = calc_b_matrix(bvals, bvecs)

    # Take the log of the data
    dwi[dwi <= 0] = np.finfo(float).eps
    dwi_log = np.log(dwi)

    # Allocate Space for Outputs
    eigen_values = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 3))
    eigen_vectors = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 3, 3))
    dk_tensor = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 3))

    # Perform Weighted Linear least squares fitting
    count = 0.0
    percent_prev = 0.0
    num_vox = np.sum(mask)

    for i in range(dwi.shape[0]):
        for j in range(dwi.shape[1]):
            for k in range(dwi.shape[2]):
                if mask[i,j,k] != 0:
                    W = np.diag(dwi[i,j,k,:]) ** 2

                    # Fit Ax = B  (b_matrix' * W * b_matrix * x = b_matrix' * W * signal)
                    x = (np.linalg.lstsq(np.dot(np.dot(b_matrix.T, W), b_matrix),
                                         np.dot(np.dot(b_matrix.T, W), dwi_log[i,j,k,:]))[0])

                    # Organize the Tensor
                    diff_tensor = np.array([[x[1], x[4], x[5]], [x[4], x[2], x[6]], [x[5], x[6], x[3]]])
                    dki = np.zeros((3,3,3,3))
                    dki[0,0,0,0] = x[7]
                    dki[0,0,0,1] = x[8]
                    dki[0,0,1,0] = x[8]
                    dki[0,1,0,0] = x[8]
                    dki[1,0,0,0] = x[8]
                    dki[0,0,1,1] = x[9]
                    dki[0,1,1,0] = x[9]
                    dki[1,1,0,0] = x[9]
                    dki[1,0,0,1] = x[9]
                    dki[1,0,1,0] = x[9]
                    dki[0,1,0,1] = x[9]
                    dki[1,1,1,0] = x[10]
                    dki[1,1,0,1] = x[10]
                    dki[1,0,1,1] = x[10]
                    dki[0,1,1,1] = x[10]
                    dki[1,1,1,1] = x[11]
                    dki[0,0,0,2] = x[12]
                    dki[0,0,2,0] = x[12]
                    dki[0,2,0,0] = x[12]
                    dki[2,0,0,0] = x[12]
                    dki[0,0,1,2] = x[13]
                    dki[0,1,0,2] = x[13]
                    dki[1,0,0,2] = x[13]
                    dki[1,0,2,0] = x[13]
                    dki[0,1,2,0] = x[13]
                    dki[0,0,2,1] = x[13]
                    dki[1,2,0,0] = x[13]
                    dki[0,2,1,0] = x[13]
                    dki[0,2,0,1] = x[13]
                    dki[2,1,0,0] = x[13]
                    dki[2,0,1,0] = x[13]
                    dki[2,0,0,1] = x[13]
                    dki[1,1,0,2] = x[14]
                    dki[1,1,2,0] = x[14]
                    dki[0,1,1,2] = x[14]
                    dki[2,1,1,0] = x[14]
                    dki[0,2,1,1] = x[14]
                    dki[2,0,1,1] = x[14]
                    dki[1,0,2,1] = x[14]
                    dki[1,2,0,1] = x[14]
                    dki[0,1,2,1] = x[14]
                    dki[2,1,0,1] = x[14]
                    dki[1,0,1,2] = x[14]
                    dki[1,2,1,0] = x[14]
                    dki[1,1,1,2] = x[15]
                    dki[1,1,2,1] = x[15]
                    dki[1,2,1,1] = x[15]
                    dki[2,1,1,1] = x[15]
                    dki[0,0,2,2] = x[16]
                    dki[0,2,0,2] = x[16]
                    dki[2,0,0,2] = x[16]
                    dki[2,2,0,0] = x[16]
                    dki[0,2,2,0] = x[16]
                    dki[2,0,2,0] = x[16]
                    dki[0,1,2,2] = x[17]
                    dki[1,0,2,2] = x[17]
                    dki[0,2,2,1] = x[17]
                    dki[1,2,2,0] = x[17]
                    dki[2,2,0,1] = x[17]
                    dki[2,2,1,0] = x[17]
                    dki[0,2,1,2] = x[17]
                    dki[1,2,0,2] = x[17]
                    dki[2,0,2,1] = x[17]
                    dki[2,1,2,0] = x[17]
                    dki[2,1,0,2] = x[17]
                    dki[2,0,1,2] = x[17]
                    dki[1,1,2,2] = x[18]
                    dki[1,2,1,2] = x[18]
                    dki[1,2,2,1] = x[18]
                    dki[2,1,2,1] = x[18]
                    dki[2,2,1,1] = x[18]
                    dki[2,1,1,2] = x[18]
                    dki[0,2,2,2] = x[19]
                    dki[2,0,2,2] = x[19]
                    dki[2,2,0,2] = x[19]
                    dki[2,2,2,0] = x[19]
                    dki[1,2,2,2] = x[20]
                    dki[2,1,2,2] = x[20]
                    dki[2,2,1,2] = x[20]
                    dki[2,2,2,1] = x[20]
                    dki[2,2,2,2] = x[21]

                    # Diagonalize the Tensor
                    eigen_values[i,j,k,:], eigen_vectors[i,j,k,:,:] = np.linalg.eig(diff_tensor)

                    # Sort the Eigenvalues
                    eigen_values[i,j,k,:], eigen_vectors[i,j,k,:,:] = dti.sort_eigvals(eigen_values[i,j,k,:], eigen_vectors[i,j,k,:])

                    # Divide by Trace**2 to get kurtosis Tensor
                    Trace = (eigen_values[i,j,k,0] + eigen_values[i,j,k,1] + eigen_values[i,j,k,2]) / 3.0
                    dki = dki / (Trace ** 2)

                    # Rotate the Kurtosis Tensor
                    dk_tensor[i,j,k,:] = rotate_kurtosis_tensor(dki, eigen_vectors[i,j,k,:,:])

                    # Scale 3 orthogonal Kurtosis elements
                    dk_tensor[i,j,k,0] = dk_tensor[i,j,k,0] * Trace**2 / (eigen_values[i,j,k,0] ** 2)
                    dk_tensor[i,j,k,1] = dk_tensor[i,j,k,1] * Trace**2 / (eigen_values[i,j,k,1] ** 2)
                    dk_tensor[i,j,k,2] = dk_tensor[i,j,k,2] * Trace**2 / (eigen_values[i,j,k,2] ** 2)

                    # Update Progress
                    count += 1.0
                    percent = np.around((count / num_vox * 100), decimals = 1)
                    if(percent != percent_prev):
                        util.progress_update("Fitting DKI: ", percent)
                        percent_prev = percent

    # Return the Eigenvalues and Eigenvectors
    return dk_tensor, eigen_values, eigen_vectors

def rotate_kurtosis_tensor(dki, ev):
    # Only storing 1111, 2222, 3333 elements of kurtosis tensor
    dk_tensor = np.zeros(3)

    for x in range(3):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        dk_tensor[x] += ev[i,x] * ev[j,x] * ev[k,x] * ev[l,x] * dki[i,j,k,l]

    return dk_tensor

def calc_dki_parameters(dki_tensor):
    AK = dki_tensor[:,:,:,2]
    RK = (dki_tensor[:,:,:,0] + dki_tensor[:,:,:,1]) / 2.0
    MK = (dki_tensor[:,:,:,0] + dki_tensor[:,:,:,1] + dki_tensor[:,:,:,2]) / 3.0
    FAK = np.sqrt(1.5 * ((dki_tensor[:,:,:,0]-MK)**2 + (dki_tensor[:,:,:,1]-MK)**2 + (dki_tensor[:,:,:,2]-MK)**2) / (dki_tensor[:,:,:,0]**2 + dki_tensor[:,:,:,1]**2 + dki_tensor[:,:,:,2]**2))

    return FAK, MK, AK, RK

def calc_b_matrix(bval, bvec):
    b_matrix = np.ones((bval.shape[0], 22))

    # DTI portion of b-matrix
    b_matrix[:, 1] = bvec[:,0] ** 2
    b_matrix[:, 2] = bvec[:,1] ** 2
    b_matrix[:, 3] = bvec[:,2] ** 2
    b_matrix[:, 4] = 2 * bvec[:,0] * bvec[:,1]
    b_matrix[:, 5] = 2 * bvec[:,0] * bvec[:,2]
    b_matrix[:, 6] = 2 * bvec[:,1] * bvec[:,2]

    for i in range(1,7):
        b_matrix[:,i] *= -bval

    # DKI portion of b-matrix
    b_matrix[:, 7] = bvec[:,0]**4
    b_matrix[:, 8] = 4 * bvec[:,0]**3 * bvec[:,1]
    b_matrix[:, 9] = 6 * bvec[:,0]**2 * bvec[:,1]**2
    b_matrix[:, 10] = 4 * bvec[:,0] * bvec[:,1]**3
    b_matrix[:, 11] = bvec[:,1]**4
    b_matrix[:, 12] = 4 * bvec[:,0]**3 * bvec[:,2]
    b_matrix[:, 13] = 12 * bvec[:,0]**2 * bvec[:,1] * bvec[:,2]
    b_matrix[:, 14] = 12 * bvec[:,0] * bvec[:,1]**2 * bvec[:,2]
    b_matrix[:, 15] = 4 * bvec[:,1]**3 * bvec[:,2]
    b_matrix[:, 16] = 6 * bvec[:,0]**2 * bvec[:,2]**2
    b_matrix[:, 17] = 12 * bvec[:,0] * bvec[:,1] * bvec[:,2]**2
    b_matrix[:, 18] = 6 * bvec[:,1]**2 * bvec[:,2]**2
    b_matrix[:, 19] = 4 * bvec[:,0] * bvec[:,2]**3
    b_matrix[:, 20] = 4 * bvec[:,1] * bvec[:,2]**3
    b_matrix[:, 21] = bvec[:,2]**4

    for i in range(7,22):
        b_matrix[:,i] *= (bval**2)/6.0

    return b_matrix

def filter_bvals(dwi, bvals, bvecs, bval_threshold=10000):

    count = 0
    for i in range(bvals.shape[0]):
        if bvals[i] <= bval_threshold:
            count += 1

    bval_thresh = np.zeros((count))
    bvec_thresh = np.zeros((count,3))
    dwi_thresh = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], count))

    index = 0
    for i in range(bvals.shape[0]):
        if bvals[i] <= bval_threshold:
            bval_thresh[index] = bvals[i]
            bvec_thresh[index,:] = bvecs[i,:]
            dwi_thresh[:,:,:,index] = dwi[:,:,:,i]
            index += 1

    return dwi_thresh, bval_thresh, bvec_thresh
