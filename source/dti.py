import numpy as np
import nibabel as nib
from . import util


def main_dti(dwi_file, bval_file, bvec_file, mask_file, out_path, b_thresh=2100,
             calc_FA=True, calc_MD=True, calc_AD=True, calc_RD=True, output_tensor=True,
             output_dec_map=True, output_eig_vals=True, output_eig_vecs=True,
             output_r_squared=True):

    # Load in data
    dwi, mask, bvals, bvecs = util.load_diffusion_data(dwi_file, bval_file, bvec_file, mask_file)
    data = dwi.get_data()
    mask = mask.get_data()

    # Filter b-values
    data, bvals, bvecs = filter_bvals(data, bvals, bvecs, b_thresh)

    # Fit the data
    eigen_values, eigen_vectors, tensor, r_squared = fit_dti(data, bvals, bvecs, mask)

    # Calculate and Save Scalar Maps
    if calc_FA:
        FA = calc_fa(eigen_values)

        # Save
        img = nib.Nifti1Image(FA, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'FA.nii'))

    if calc_MD or calc_AD or calc_RD:
        MD, AD, RD = calc_diffusivities(eigen_values)
        # Save
        if calc_MD:
            img = nib.Nifti1Image(MD, dwi.affine, dwi.header)
            nib.save(img, (out_path + 'MD.nii'))
        if calc_AD:
            img = nib.Nifti1Image(AD, dwi.affine, dwi.header)
            nib.save(img, (out_path + 'AD.nii'))
        if calc_RD:
            img = nib.Nifti1Image(RD, dwi.affine, dwi.header)
            nib.save(img, (out_path + 'RD.nii'))

    if output_dec_map:
        dec = np.zeros(eigen_values.shape)
        dec[:,:,:,0] = eigen_vectors[:,:,:,0,2] * FA
        dec[:,:,:,1] = eigen_vectors[:,:,:,1,2] * FA
        dec[:,:,:,2] = eigen_vectors[:,:,:,2,2] * FA

        img = nib.Nifti1Image(dec, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'DECmap.nii'))

    if output_eig_vals:
        img = nib.Nifti1Image(eigen_values[:,:,:,2], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'EigenValues_1.nii'))
        img = nib.Nifti1Image(eigen_values[:,:,:,1], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'EigenValues_2.nii'))
        img = nib.Nifti1Image(eigen_values[:,:,:,0], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'EigenValues_3.nii'))

    if output_eig_vecs:
        img = nib.Nifti1Image(eigen_vectors[:,:,:,:,0], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'EigenVector_3.nii'))
        img = nib.Nifti1Image(eigen_vectors[:,:,:,:,1], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'EigenVector_2.nii'))
        img = nib.Nifti1Image(eigen_vectors[:,:,:,:,2], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'EigenVector_1.nii'))

    if output_r_squared:
        img = nib.Nifti1Image(r_squared, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'r_squared.nii'))

    if output_tensor:
        img = nib.Nifti1Image(tensor, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'DiffusionTensor.nii'))

    return eigen_values, eigen_vectors, tensor

def fit_dti(dwi, bvals, bvecs, mask):
    # Calculate the b-matrix
    b_matrix = calc_b_matrix(bvals, bvecs)

    # Take the log of the data
    dwi[dwi <= 0] = np.finfo(float).eps
    dwi_log = np.log(dwi)

    # Allocate Space for Outputs
    eigen_values = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 3))
    eigen_vectors = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 3, 3))

    # Perform Weighted Linear least squares fitting
    count = 0.0
    percent_prev = 0.0
    num_vox = np.sum(mask)

    tensor_img = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 6))
    r_squared = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2]))
    for i in range(dwi.shape[0]):
        for j in range(dwi.shape[1]):
            for k in range(dwi.shape[2]):
                if mask[i,j,k] != 0:
                    W = np.diag(dwi[i,j,k,:]) ** 2

                    # Fit Ax = B  (b_matrix' * W * b_matrix * x = b_matrix' * W * signal)
                    x = (np.linalg.lstsq(np.matmul(np.matmul(b_matrix.T, W), b_matrix), np.matmul(np.matmul(b_matrix.T, W), dwi_log[i,j,k,:]), rcond=None)[0])

                    # Coefficient of Determination
                    predicted_vals = np.matmul(b_matrix, x)

                    rho = np.corrcoef(predicted_vals, dwi_log[i,j,k,:])

                    r_squared[i,j,k] = rho[0][1] ** 2

                    # Organize the Tensor
                    tensor = np.array([[x[1], x[4], x[5]],
                                       [x[4], x[2], x[6]],
                                       [x[5], x[6], x[3]]])

                    tensor_img[i,j,k,:] = [x[1], x[2], x[3], x[4], x[5], x[6]]

                    # Diagonalize the Tensor
                    eigen_values[i,j,k,:], eigen_vectors[i,j,k,:,:] = np.linalg.eig(tensor)

                    # Sort the Eigenvalues
                    eigen_values[i,j,k,:], eigen_vectors[i,j,k,:,:] = sort_eigvals(eigen_values[i,j,k,:], eigen_vectors[i,j,k,:])

                    # Update Progress
                    count += 1.0
                    percent = np.around((count / num_vox * 100), decimals = 1)
                    if(percent != percent_prev):
                        util.progress_update("Fitting DTI: ", percent)
                        percent_prev = percent

    # Return the Eigenvalues and Eigenvectors
    return eigen_values, eigen_vectors, tensor_img, r_squared

def calc_fa(eigen_values):
    numerator = (np.sqrt(1/2.0) * np.sqrt((eigen_values[:,:,:,0] - eigen_values[:,:,:,1]) ** 2 +
                 (eigen_values[:,:,:,1] - eigen_values[:,:,:,2]) ** 2 +
                 (eigen_values[:,:,:,2] - eigen_values[:,:,:,0]) ** 2))

    denominator = (np.sqrt(eigen_values[:,:,:,0] ** 2 + eigen_values[:,:,:,1] ** 2 +
                           eigen_values[:,:,:,2] ** 2))

    denominator[denominator == 0] = 1
    numerator[denominator == 0] = 0

    FA = numerator / denominator

    return FA

def calc_diffusivities(eigen_values):
    MD = (eigen_values[:,:,:,0] + eigen_values[:,:,:,1] + eigen_values[:,:,:,2]) / 3.0
    AD = eigen_values[:,:,:,2]
    RD = (eigen_values[:,:,:,0] + eigen_values[:,:,:,1]) / 2.0

    return MD, AD, RD

def calc_b_matrix(bvals, bvecs):
    b_matrix = np.ones((bvals.shape[0], 7))
    b_matrix[:, 1] = bvecs[:,0] ** 2
    b_matrix[:, 2] = bvecs[:,1] ** 2
    b_matrix[:, 3] = bvecs[:,2] ** 2
    b_matrix[:, 4] = 2 * bvecs[:,0] * bvecs[:,1]
    b_matrix[:, 5] = 2 * bvecs[:,0] * bvecs[:,2]
    b_matrix[:, 6] = 2 * bvecs[:,1] * bvecs[:,2]

    for i in range(1,7):
        b_matrix[:,i] *= -bvals

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

def sort_eigvals(eigen_values, eigen_vectors):
    eig_vecs_sorted = np.zeros((3,3))

    eig_vals_sorted = np.sort(eigen_values)

    for i in range(3):
        for j in range(3):
            if(eig_vals_sorted[i] == eigen_values[j]):
                eig_vecs_sorted[:,i] = eigen_vectors[:,j]

    return eig_vals_sorted, eig_vecs_sorted
