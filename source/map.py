import numpy as np
import nibabel as nib
import scipy.special
from . import util
from . import dti
from . import spherical_harmonics as SH
import os

def main_map(dwi_file, bval_file, bvec_file, mask_file, little_delta, big_delta,
             out_path, order=6, SH_order=8, b_thresh_dti=2100, calc_rtps=True,
             calc_ng=True, calc_pa=True, calc_dki=False, return_dti=False,
             return_glyphs=True):

    # Load in Data
    dwi, mask, bvals, bvecs = util.load_diffusion_data(dwi_file, bval_file, bvec_file, mask_file)
    data = dwi.get_data()
    mask = mask.get_data()

    # Fit DTI
    if return_dti:
        eigen_values, eigen_vectors, tensor = dti.main_dti(dwi_file, bval_file, bvec_file,
                            mask_file, (out_path + "DTI_"), b_thresh_dti, True, True, True, True, True, True, True, True)
        eigen_values[eigen_values <= 0] = 1e-5
    else:
        eigen_values, eigen_vectors, tensor = dti.main_dti(dwi_file, bval_file, bvec_file,
                            mask_file, "", b_thresh_dti, False, False, False, False, False, False, False, False)
        eigen_values[eigen_values <= 0] = 1e-5

    # Determine Diffusion Time
    diffusion_time = big_delta - little_delta / 3

    # Calculate u-vectors
    uvectors = np.sqrt(2 * eigen_values * diffusion_time)

    # Convert b values and vectors to q-vectors
    qvectors = util.b_to_q(bvals, bvecs, big_delta, little_delta)

    # Invert Eigenvectors (Inverse of a rotation matrix is its transpose)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                eigen_vectors[i,j,k,:,:] = eigen_vectors[i,j,k,:,:].T

    # Fit MAP
    coeffs = fit_map(data, qvectors, mask, diffusion_time, uvectors, eigen_vectors,
                     order, lam=0.2)

    num_coeffs = int(round(1.0/6 * (order/2 + 1) * (order/2 + 2) * (2*order + 3)))
    uvectors = np.sort(uvectors, axis=3)


    ### WRITE OUTPUT ###########################################################
    # Save Coefficients and UVectors
    img = nib.Nifti1Image(coeffs, dwi.affine, dwi.header)
    nib.save(img, (out_path + 'coeffs.nii'))
    img = nib.Nifti1Image(uvectors, dwi.affine, dwi.header)
    nib.save(img, (out_path + 'uvecs.nii'))

    # Calculate and Save Scalar MAPs
    if calc_pa:
        pa_dti, pa_dti_theta, pa, pa_theta, pa_jsd = calc_propagator_anisotropy(coeffs, uvectors, order, mask)

        img = nib.Nifti1Image(pa_dti, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'PA_DTI.nii'))

        #img = nib.Nifti1Image(pa_dti_theta, dwi.affine, dwi.header)
        #nib.save(img, (out_path + 'PA_DTI_theta.nii'))

        img = nib.Nifti1Image(pa, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'PA.nii'))

        #img = nib.Nifti1Image(pa_theta, dwi.affine, dwi.header)
        #nib.save(img, (out_path + 'PA_theta.nii'))

        #img = nib.Nifti1Image(pa_jsd, dwi.affine, dwi.header)
        #nib.save(img, (out_path + 'PA_JSD.nii'))

    if calc_ng:
        print("Calculating Non-Gaussianity")
        ng, ng_par, ng_perp, ng_jsd = calc_non_gaussianity(coeffs, order, num_coeffs)

        img = nib.Nifti1Image(ng, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'NG.nii'))

        img = nib.Nifti1Image(ng_par[:,:,:,2], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'AxialNG_1.nii'))

        img = nib.Nifti1Image(ng_par[:,:,:,1], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'AxialNG_2.nii'))

        img = nib.Nifti1Image(ng_par[:,:,:,0], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'AxialNG_3.nii'))

        img = nib.Nifti1Image(ng_perp[:,:,:,2], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RadialNG_1.nii'))

        img = nib.Nifti1Image(ng_perp[:,:,:,1], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RadialNG_2.nii'))

        img = nib.Nifti1Image(ng_perp[:,:,:,0], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RadialNG_3.nii'))

        #img = nib.Nifti1Image(ng_jsd, dwi.affine, dwi.header)
        #nib.save(img, (out_path + 'NG_JSD.nii'))

    if calc_rtps:
        print("Calculating Return origin, axis, and plane probabilities")
        rtop, rtap, rtpp = calc_return_to_probabilities(coeffs, uvectors, order, num_coeffs)

        img = nib.Nifti1Image(rtop, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RTOP.nii'))

        img = nib.Nifti1Image(rtap[:,:,:,2], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RTAP_1.nii'))

        img = nib.Nifti1Image(rtap[:,:,:,1], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RTAP_2.nii'))

        img = nib.Nifti1Image(rtap[:,:,:,0], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RTAP_3.nii'))

        img = nib.Nifti1Image(rtpp[:,:,:,2], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RTPP_1.nii'))

        img = nib.Nifti1Image(rtpp[:,:,:,1], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RTPP_2.nii'))

        img = nib.Nifti1Image(rtpp[:,:,:,0], dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RTPP_3.nii'))

    if calc_dki:
        print("Calculating DKI parameters from MAP Coefficients")
        mk, k_par, k_perp, fa_k = calc_gdti_params(coeffs, uvectors, mask, little_delta, big_delta, order, moment_order=4)

        img = nib.Nifti1Image(mk, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'MK_MAP.nii'))

        img = nib.Nifti1Image(k_par, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'AK_MAP.nii'))

        img = nib.Nifti1Image(k_perp, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'RK_MAP.nii'))

        img = nib.Nifti1Image(fa_k, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'KA_MAP.nii'))

    if return_dti:
        img = nib.Nifti1Image(eigen_vectors, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'dti_eigen_vectors.nii'))

        img = nib.Nifti1Image(eigen_values, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'dti_eigen_values.nii'))

        img = nib.Nifti1Image(tensor, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'DiffusionTensor.nii'))

    if return_glyphs:
        MAP_Glyphs = fit_map_glyphs(coeffs, uvectors, eigen_vectors, order, mask, moment=2)

        img = nib.Nifti1Image(MAP_Glyphs, dwi.affine, dwi.header)
        nib.save(img, (out_path + 'MAP_ODFs.nii'))
    ############################################################################
    return coeffs

def fit_map(data, qvectors, mask, diffusion_time, uvectors, eigen_vectors,
            order = 6, lam = 0.2):

    # Calculate the number of coefficients for a give order
    num_coeffs = int(round(1.0/6 * (order/2 + 1) * (order/2 + 2) * (2*order + 3)))

    # Allocate space for Outputs
    data[data <= 0] = np.finfo(float).eps
    coeffs = np.zeros((data.shape[0], data.shape[1], data.shape[2], num_coeffs))

    # Calculate "B" for estimating B0
    B = calc_b(order, num_coeffs)

    # Calculate Coefficients for Regularization Matrix
    reg_matrix_coeffs = calc_reg_matrix_coeffs(order, num_coeffs)

    count = 0.0
    percent_prev = 0.0
    num_vox = np.sum(mask)

    for i in range(data.shape[0]):
        for j in range (data.shape[1]):
            for k in range(data.shape[2]):
                if mask[i,j,k] != 0:
                    # Rotate qvectors
                    qvec = np.dot(eigen_vectors[i,j,k,:,:], qvectors.T).T

                    # Evaluate Gaussian Hermite Basis Functions
                    Q = gaussian_hermite_q_space(order, num_coeffs, qvec, uvectors[i,j,k,:])

                    # Create Regularization for least squares fit
                    R = create_regularization_matrix(order, num_coeffs, uvectors[i,j,k,:], reg_matrix_coeffs)

                    # Perform Linear Fit
                    coeffs[i,j,k,:] = (np.linalg.lstsq((np.dot(Q.T, Q) + lam * R), (np.dot(Q.T, data[i,j,k,:])), rcond=None)[0])

                    # Normalize coefficients by estimated B0
                    EstimatedB0 = np.dot(coeffs[i,j,k,:], B)
                    coeffs[i,j,k,:] /= EstimatedB0

                    # Update Progress
                    count += 1.0
                    percent = np.around((count / num_vox * 100), decimals = 1)
                    if(percent != percent_prev):
                        util.progress_update("Fitting MAP: ", percent)
                        percent_prev = percent

    return coeffs

def gaussian_hermite_q_space(order, num_coeffs, qvec, uvector):
    # Allocate Space for output
    Q = np.zeros((qvec.shape[0], num_coeffs), dtype=complex)
    H = np.zeros((qvec.shape[0], order+1, 3), dtype=complex)

    # Points to evaluate hermite polynomials at
    x1 = 2 * np.pi * uvector[0] * qvec[:,0]
    x2 = 2 * np.pi * uvector[1] * qvec[:,1]
    x3 = 2 * np.pi * uvector[2] * qvec[:,2]

    # Evaluate separated basis functions at each order
    for i in range(order+1):
        coeffs = np.zeros(order+1)
        coeffs[i] = 1

        H[:,i,0] = (1j**(-i) / np.sqrt(2**i * np.math.factorial(i)) *
                    np.exp(-(x1**2)/2) * np.polynomial.hermite.hermval(x1, coeffs))
        H[:,i,1] = (1j**(-i) / np.sqrt(2**i * np.math.factorial(i)) *
                    np.exp(-(x2**2)/2) * np.polynomial.hermite.hermval(x2, coeffs))
        H[:,i,2] = (1j**(-i) / np.sqrt(2**i * np.math.factorial(i)) *
                    np.exp(-(x3**2)/2) * np.polynomial.hermite.hermval(x3, coeffs))

    # Multiply separable functions together to get 3-D function
    index = 0
    for N in range(0,order+1,2):
        for n1 in range(N+1):
            for n2 in range(N+1):
                for n3 in range(N+1):
                    if((n1+n2+n3) == N):
                        Q[:,index] = H[:,n1,0] * H[:,n2,1] * H[:,n3,2]
                        index += 1

    return np.real(Q)

def calc_b(order, num_coeffs):
    # Allocate Space
    B = np.zeros(num_coeffs)

    index = 0
    for N in range(0,order+1,2):
        for n1 in range(order+1):
            for n2 in range(order+1):
                for n3 in range(order+1):
                    if((n1+n2+n3) == N):
                        if((n1 % 2) == 0):
                            B1 = np.sqrt(util.factn(n1,1)) / util.factn(n1,2)
                            B2 = np.sqrt(util.factn(n2,1)) / util.factn(n2,2)
                            B3 = np.sqrt(util.factn(n3,1)) / util.factn(n3,2)

                            B[index] = B1 * B2 * B3
                            index += 1

    return B

def create_regularization_matrix(order, num_coeffs, uvector, reg_matrix_coeffs):
    R = np.zeros((num_coeffs, num_coeffs))
    scale = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    ux = uvector[0]
    uy = uvector[1]
    uz = uvector[2]

    scale[0] = ux ** 3 / (uy * uz)
    scale[1] = 2 * ux * uy / uz
    scale[2] = uy ** 3 / (uz * ux)
    scale[3] = 2 * uy * uz / ux
    scale[4] = uz ** 3 / (ux * uy)
    scale[5] = 2 * ux * uz / uy

    for i in range(6):
        R = R + scale[i] * reg_matrix_coeffs[:,:,i]

    return R

def calc_reg_matrix_coeffs(order, num_coeffs):
    reg_matrix_coeffs = np.zeros((num_coeffs, num_coeffs, 6))

    U = calc_u_matrix(order)
    T = calc_t_matrix(order)
    S = calc_s_matrix(order)

    indexN = 0
    for N in range(0,order+1,2):
        for n1 in range(order+1):
            for n2 in range(order+1):
                for n3 in range(order+1):
                    if((n1+n2+n3) == N):
                        indexM = 0

                        for M in range(0,order+1,2):
                            for m1 in range(order+1):
                                for m2 in range(order+1):
                                    for m3 in range(order+1):
                                        if((m1+m2+m3) == M):
                                            reg_matrix_coeffs[indexN, indexM, 0] = S[n1,m1] * U[n2,m2] * U[n3,m3]
                                            reg_matrix_coeffs[indexN, indexM, 1] = T[n1,m1] * T[n2,m2] * U[n3,m3]
                                            reg_matrix_coeffs[indexN, indexM, 2] = U[n1,m1] * S[n2,m2] * U[n3,m3]
                                            reg_matrix_coeffs[indexN, indexM, 3] = U[n1,m1] * T[n2,m2] * T[n3,m3]
                                            reg_matrix_coeffs[indexN, indexM, 4] = U[n1,m1] * U[n2,m2] * S[n3,m3]
                                            reg_matrix_coeffs[indexN, indexM, 5] = T[n1,m1] * U[n2,m2] * T[n3,m3]

                                            indexM += 1
                        indexN += 1

    return reg_matrix_coeffs

def calc_u_matrix(order):
    U = np.zeros((order+1, order+1))

    for n in range(order+1):
        for m in range(order+1):
            if(m == n):
                U[n,m] = (-1) ** n / (2 * np.sqrt(np.pi))

    return U

def calc_t_matrix(order):
    T = np.zeros((order+1, order+1))
    mn_factors = [0.0, 0.0, 0.0]

    for n in range(order+1):
        for m in range(order+1):
            if(m == n):
                mn_factors[0] = (1+2*n)
            else:
                mn_factors[0] = 0

            if((m+2) == n):
                mn_factors[1] = np.sqrt((n * (n-1)))
            else:
                mn_factors[1] = 0

            if((m == (n+2))):
                mn_factors[2] = np.sqrt((m * (m-1)))
            else:
                mn_factors[2] = 0

            T[n,m] = (-1) ** (n+1) * np.pi ** (1.5) * np.sum(mn_factors)

    return T

def calc_s_matrix(order):
    S = np.zeros((order+1, order+1))
    mn_factors = [0.0, 0.0, 0.0, 0.0, 0.0]

    for n in range(order+1):
        for m in range(order+1):
            if(m == n):
                mn_factors[0] = 3 * (2*n**2 + 2*n + 1)
            else:
                mn_factors[0] = 0

            if(m == (n+2)):
                mn_factors[1] = (6+4*n) * np.sqrt((util.factn(m,1) / util.factn(n,1)))
            else:
                mn_factors[1] = 0

            if(m == (n+4)):
                mn_factors[2] = np.sqrt((util.factn(m,1) / util.factn(n,1)))
            else:
                mn_factors[2] = 0

            if((m+2) == n):
                mn_factors[3] = (6+4*m) * np.sqrt((util.factn(n,1) / util.factn(m,1)))
            else:
                mn_factors[3] = 0

            if((m+4) == n):
                mn_factors[4] = np.sqrt((util.factn(n,1) / util.factn(m,1)))
            else:
                mn_factors[4] = 0

            S[n,m] = 2*(-1)**n * np.pi**3.5 * np.sum(mn_factors)

    return S

def calc_return_to_probabilities(coeffs, uvectors, order, num_coeffs):
    # Calculate sign factors
    b, signs = calc_b_and_signs(order, num_coeffs)
    b = b[0,:] * b[1,:] * b[2,:]

    # Reshape Data for Faster Processing
    uvectors_lin = np.reshape(uvectors, (uvectors.shape[0] * uvectors.shape[1] * uvectors.shape[2], uvectors.shape[3]))
    coeffs_lin = np.reshape(coeffs, (coeffs.shape[0] * coeffs.shape[1] * coeffs.shape[2], coeffs.shape[3]))

    # Calculate Coefficients
    rtop_scale = (8 * np.pi**3 * uvectors_lin[:,0]**2 * uvectors_lin[:,1]**2 * uvectors_lin[:,2]**2) ** (-0.5)
    rtap_scale_x = (4 * np.pi**2 * uvectors_lin[:,1]**2 * uvectors_lin[:,2]**2) ** (-0.5)
    rtap_scale_y = (4 * np.pi**2 * uvectors_lin[:,0]**2 * uvectors_lin[:,2]**2) ** (-0.5)
    rtap_scale_z = (4 * np.pi**2 * uvectors_lin[:,0]**2 * uvectors_lin[:,1]**2) ** (-0.5)
    rtpp_scale_x = (2 * np.pi * uvectors_lin[:,0]**2)**(-0.5)
    rtpp_scale_y = (2 * np.pi * uvectors_lin[:,1]**2)**(-0.5)
    rtpp_scale_z = (2 * np.pi * uvectors_lin[:,2]**2)**(-0.5)

    # Estimate Probabilites
    rtop = np.zeros(coeffs_lin.shape[0])
    rtap_x = np.zeros(coeffs_lin.shape[0])
    rtap_y = np.zeros(coeffs_lin.shape[0])
    rtap_z = np.zeros(coeffs_lin.shape[0])
    rtpp_x = np.zeros(coeffs_lin.shape[0])
    rtpp_y = np.zeros(coeffs_lin.shape[0])
    rtpp_z = np.zeros(coeffs_lin.shape[0])
    rtap = np.zeros((coeffs.shape[0], coeffs.shape[1], coeffs.shape[2], 3))
    rtpp = np.zeros((coeffs.shape[0], coeffs.shape[1], coeffs.shape[2], 3))

    rtop = rtop_scale * np.dot(signs[0,:]*signs[1,:]*signs[2,:]*b, coeffs_lin.T)
    rtap_x = rtap_scale_x * np.dot(signs[1,:]*signs[2,:]*b, coeffs_lin.T)
    rtap_y = rtap_scale_y * np.dot(signs[0,:]*signs[2,:]*b, coeffs_lin.T)
    rtap_z = rtap_scale_z * np.dot(signs[0,:]*signs[1,:]*b, coeffs_lin.T)
    rtpp_x = rtpp_scale_x * np.dot(signs[0,:]*b, coeffs_lin.T)
    rtpp_y = rtpp_scale_y * np.dot(signs[1,:]*b, coeffs_lin.T)
    rtpp_z = rtpp_scale_z * np.dot(signs[2,:]*b, coeffs_lin.T)


    # Filter Data for Crazy Outliers
    rtop[rtop < 0] = 0
    rtap_x[rtap_x < 0] = 0
    rtap_y[rtap_y < 0] = 0
    rtap_z[rtap_z < 0] = 0
    rtpp_x[rtpp_x < 0] = 0
    rtpp_y[rtpp_y < 0] = 0
    rtpp_z[rtpp_z < 0] = 0

    rtop = rtop ** (1.0/3)
    rtap_x = rtap_x ** 0.5
    rtap_y = rtap_y ** 0.5
    rtap_z = rtap_z ** 0.5

    rtop = np.reshape(rtop, coeffs.shape[0:3])
    rtap[:,:,:,0] = np.reshape(rtap_x, coeffs.shape[0:3])
    rtap[:,:,:,1] = np.reshape(rtap_y, coeffs.shape[0:3])
    rtap[:,:,:,2] = np.reshape(rtap_z, coeffs.shape[0:3])
    rtpp[:,:,:,0] = np.reshape(rtpp_x, coeffs.shape[0:3])
    rtpp[:,:,:,1] = np.reshape(rtpp_y, coeffs.shape[0:3])
    rtpp[:,:,:,2] = np.reshape(rtpp_z, coeffs.shape[0:3])

    rtop = np.real(rtop)
    rtap = np.real(rtap)
    rtpp = np.real(rtpp)

    rtop[np.isnan(rtop)] = 0
    rtap[np.isnan(rtap)] = 0
    rtpp[np.isnan(rtpp)] = 0

    return rtop, rtap, rtpp


def calc_non_gaussianity(coeffs, order, num_coeffs):
    # Calculate sign factors
    b, signs = calc_b_and_signs(order, num_coeffs)

    #Reshape for Faster Processing
    coeffs_lin = np.reshape(coeffs, (coeffs.shape[0] * coeffs.shape[1] * coeffs.shape[2], num_coeffs))
    a1 = np.zeros((coeffs_lin.shape[0], order+1))
    a2 = np.zeros((coeffs_lin.shape[0], order+1))
    a3 = np.zeros((coeffs_lin.shape[0], order+1))
    a23 = np.zeros((coeffs_lin.shape[0], (order+1)**2))
    a13 = np.zeros((coeffs_lin.shape[0], (order+1)**2))
    a12 = np.zeros((coeffs_lin.shape[0], (order+1)**2))
    perp_23 = np.zeros((coeffs_lin.shape[0], num_coeffs))
    perp_13 = np.zeros((coeffs_lin.shape[0], num_coeffs))
    perp_12 = np.zeros((coeffs_lin.shape[0], num_coeffs))

    ng = np.zeros(coeffs_lin.shape[0])
    ng_par_x = np.zeros(coeffs_lin.shape[0])
    ng_par_y = np.zeros(coeffs_lin.shape[0])
    ng_par_z = np.zeros(coeffs_lin.shape[0])
    ng_perp_x = np.zeros(coeffs_lin.shape[0])
    ng_perp_y = np.zeros(coeffs_lin.shape[0])
    ng_perp_z = np.zeros(coeffs_lin.shape[0])

    ng_par = np.zeros((coeffs.shape[0], coeffs.shape[1], coeffs.shape[2], 3))
    ng_perp = np.zeros((coeffs.shape[0], coeffs.shape[1], coeffs.shape[2], 3))

    # Parallel Coefficients
    index = 0
    for N in range(0,order+1,2):
        for n1 in range(N+1):
            for n2 in range(N+1):
                for n3 in range(N+1):
                    if((n1+n2+n3) == N):
                        a1[:,n1] += signs[1,index] * signs[2,index] * b[1,index] * b[2,index] * coeffs_lin[:,index]
                        a2[:,n2] += signs[0,index] * signs[2,index] * b[0,index] * b[2,index] * coeffs_lin[:,index]
                        a3[:,n3] += signs[0,index] * signs[1,index] * b[0,index] * b[1,index] * coeffs_lin[:,index]
                        index += 1

    # Perpindicular Coefficients
    n1_index = []
    n2_index = []
    n3_index = []
    index = 0
    for N in range(0,order+1,2):
        for n1 in range(N+1):
            for n2 in range(N+1):
                for n3 in range(N+1):
                    if((n1+n2+n3) == N):
                        perp_23[:,index] = signs[0,index] * b[0,index] * coeffs_lin[:,index]
                        perp_13[:,index] = signs[1,index] * b[1,index] * coeffs_lin[:,index]
                        perp_12[:,index] = signs[2,index] * b[2,index] * coeffs_lin[:,index]
                        n1_index.append(n1);
                        n2_index.append(n2);
                        n3_index.append(n3);
                        index += 1

    index = 0
    for N in range(order+1):
        for M in range(order+1):

            for i in range(num_coeffs):
                if (n2_index[i] == N and n3_index[i] == M):
                    a23[:,index] += perp_23[:,i]

            for i in range(num_coeffs):
                if (n1_index[i] == N and n3_index[i] == M):
                    a13[:,index] += perp_13[:,i]

            for i in range(num_coeffs):
                if (n1_index[i] == N and n2_index[i] == M):
                    a12[:,index] += perp_12[:,i]

            index += 1

    # Calculate Non Gaussianity based on Jensen-Shannon Divergence
    ng_jsd = np.zeros((coeffs.shape[0], coeffs.shape[1], coeffs.shape[2]))
    coeffs_gaussian = np.zeros((num_coeffs))
    rmn = calc_rmn(order)
    for x in range(ng_jsd.shape[0]):
        for y in range(ng_jsd.shape[1]):
            for z in range(ng_jsd.shape[2]):
                coeffs_total = coeffs[x,y,z,:]
                coeffs_gaussian[0] = coeffs[x,y,z,0]
                ng_jsd[x,y,z] = np.dot(np.dot(coeffs_total-coeffs_gaussian, rmn), coeffs_total-coeffs_gaussian)


    # Calculate Nongaussianities
    with np.errstate(divide='ignore', invalid='ignore'):
        ng = np.sqrt(1 - coeffs_lin[:,0]**2 / np.sum(coeffs_lin**2, axis=1))
        ng_par_x = np.sqrt(1 - a1[:,0]**2 / np.sum(a1**2, axis = 1))
        ng_par_y = np.sqrt(1 - a2[:,0]**2 / np.sum(a2**2, axis = 1))
        ng_par_z = np.sqrt(1 - a3[:,0]**2 / np.sum(a3**2, axis = 1))
        ng_perp_x = np.sqrt(1 - a23[:,0]**2 / np.sum(a23**2, axis = 1))
        ng_perp_y = np.sqrt(1 - a13[:,0]**2 / np.sum(a13**2, axis = 1))
        ng_perp_z = np.sqrt(1 - a12[:,0]**2 / np.sum(a12**2, axis = 1))

    # Change Back to Image Matrix
    ng = np.reshape(ng, coeffs.shape[0:3])
    ng_par[:,:,:,0] = np.reshape(ng_par_x, coeffs.shape[0:3])
    ng_par[:,:,:,1] = np.reshape(ng_par_y, coeffs.shape[0:3])
    ng_par[:,:,:,2] = np.reshape(ng_par_z, coeffs.shape[0:3])
    ng_perp[:,:,:,0] = np.reshape(ng_perp_x, coeffs.shape[0:3])
    ng_perp[:,:,:,1] = np.reshape(ng_perp_y, coeffs.shape[0:3])
    ng_perp[:,:,:,2] = np.reshape(ng_perp_z, coeffs.shape[0:3])

    # Clean up and div by zeros
    ng[np.isnan(ng)] = 0
    ng_par[np.isnan(ng_par)] = 0
    ng_perp[np.isnan(ng_perp)] = 0

    return ng, ng_par, ng_perp, ng_jsd


def calc_b_and_signs(order, num_coeffs):
    b = np.zeros((3,num_coeffs))
    signs = np.zeros((3,num_coeffs))

    index = 0
    for N in range(0,order+1,2):
        for n1 in range(N+1):
            for n2 in range(N+1):
                for n3 in range(N+1):
                    if((n1+n2+n3) == N):
                        if((n1 % 2) == 0):
                            b[0,index] = np.sqrt(util.factn(n1,1)) / util.factn(n1,2)
                            signs[0,index] = (-1)**(n1/2)
                        if((n2 % 2) == 0):
                            b[1,index] = np.sqrt(util.factn(n2,1)) / util.factn(n2,2)
                            signs[1,index] = (-1)**(n2/2)
                        if((n3 % 2) == 0):
                            b[2,index] = np.sqrt(util.factn(n3,1)) / util.factn(n3,2)
                            signs[2,index] = (-1)**(n3/2)

                        index += 1

    return b, signs


def calc_propagator_anisotropy(coeffs, uvectors, order, mask):
    # Calculate Isotropic Coefficients
    u0 = calc_u0(uvectors)

    # Caluclate the Anisotropy of the DTI signal
    pa_dti, pa_dti_theta = calculate_pa_dti(u0, uvectors, mask)

    # Allocate space for all 3 outputs
    pa = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))
    pa_theta = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))
    pa_jsd = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))


    count = 0.0
    percent_prev = 0.0
    num_vox = np.sum(mask)

    # Calculate the General Propagator Anisotropy
    rmn = calc_rmn(order)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                if(mask[x,y,z] != 0):
                    # Estimate Transformation Matrix
                    tmn = calc_tmn(uvectors[x,y,z,:], u0[x,y,z], order, order/2)

                    # Calculate PA
                    pa[x,y,z], pa_theta[x,y,z], pa_jsd[x,y,z] = calculate_pa(coeffs[x,y,z,:], uvectors[x,y,z,:], u0[x,y,z], tmn, rmn)

                    # Update Progress
                    count += 1.0
                    percent = np.around((count / num_vox * 100), decimals = 1)
                    if(percent != percent_prev):
                        util.progress_update("Calculating PA: ", percent)
                        percent_prev = percent

    # Apply scaling function
    pa_dti = pa_scaling(pa_dti, 0.4)
    pa = pa_scaling(pa, 0.4)

    return pa_dti, pa_dti_theta, pa, pa_theta, pa_jsd


def calc_u0(uvectors):
    # Compute factors for polynomial
    uvectors = uvectors ** 2
    u3 = -3
    u2 = -np.sum(uvectors, axis=3)
    u1 = (uvectors[:,:,:,0] * uvectors[:,:,:,1] + uvectors[:,:,:,0] * uvectors[:,:,:,2] +
          uvectors[:,:,:,1] * uvectors[:,:,:,2])
    u = 3 * uvectors[:,:,:,0] * uvectors[:,:,:,1] * uvectors[:,:,:,2]

    # Allocate Space for Output
    u0 = np.zeros((uvectors.shape[0], uvectors.shape[1], uvectors.shape[2]))
    for x in range(uvectors.shape[0]):
        for y in range(uvectors.shape[1]):
            for z in range(uvectors.shape[2]):
                poly = np.array([u3, u2[x,y,z], u1[x,y,z], u[x,y,z]])

                # Find roots of polynomial
                r = np.roots(poly)

                # Find the single real, positive root
                for i in r:
                    if(i >= 0 and np.isreal(i)):
                        u0[x,y,z] = np.sqrt(np.real(i))
                        break

    return u0


def calculate_pa_dti(u0, uvectors, mask):
    pa_dti = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))
    pa_dti_theta = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))

    for x in range(u0.shape[0]):
        for y in range(u0.shape[1]):
            for z in range(u0.shape[2]):
                if(mask[x,y,z] != 0):
                    pa_dti[x,y,z] = ((8 * u0[x,y,z]**3 * uvectors[x,y,z,0] * uvectors[x,y,z,1] * uvectors[x,y,z,2]) /
                                     ((uvectors[x,y,z,0]**2 + u0[x,y,z]**2) * (uvectors[x,y,z,1]**2 + u0[x,y,z]**2) * (uvectors[x,y,z,2]**2 + u0[x,y,z]**2)))

                    pa_dti_theta[x,y,z] = np.arccos(np.sqrt(pa_dti[x,y,z]))
                    pa_dti[x,y,z] = np.real(np.sin(pa_dti_theta[x,y,z]))


    return pa_dti, pa_dti_theta


def calculate_pa(coeffs, uvectors, u0, tmn, rmn):

    b, norm_factor = change_basis(coeffs, tmn, u0)

    anorm =  1 / (8 * np.pi**(1.5) * uvectors[0] * uvectors[1] * uvectors[2])

    cosPA = np.sqrt(np.dot(b**2, norm_factor) / (np.sum(coeffs**2) * anorm))

    pa_theta = np.arccos(cosPA)
    pa = np.sin(pa_theta)

    # Calculate PA based on Jensen-Shannon Divergence
    b = sphere_coeffs_to_cartesian(b)
    pa_jsd = np.dot(np.dot(coeffs-b, rmn), coeffs-b)

    return pa, pa_theta, pa_jsd

def change_basis(coeffs, tmn, u0):
    norm_factor = np.zeros((tmn.shape[1]))

    for i in range(tmn.shape[1]):
        norm_factor[i] = scipy.special.gamma(2*i + 1.5) / (util.factn(2*i, 1) * 4 * np.pi**2 * u0**3)

    b = np.dot(coeffs,tmn) / norm_factor

    return b, norm_factor

def sphere_coeffs_to_cartesian(sphere_coeffs):
    order = (sphere_coeffs.shape[0] - 1) * 2
    coeffs_cart = np.zeros(int(np.round(1/6.0 * (order/2 + 1) * (order/2 + 2) * (2*order + 3))))

    index = 0
    for N in range(sphere_coeffs.shape[0]):
        for n1 in range(N*2+1):
            for n2 in range(N*2+1):
                for n3 in range(N*2+1):
                    if((n1+n2+n3) == (N*2) and (n1%2) == 0 and (n2%2) == 0 and (n3%2) == 0):
                        coeffs_cart[index] = sphere_coeffs[N] * np.sqrt(util.factn(n1,1) * util.factn(n2,1) * util.factn(n3,1)) / (util.factn(n1,2) * util.factn(n2,2) * util.factn(n3,2))
                        index += 1

    return coeffs_cart

def calc_rmn(order):
    num_coeffs = int(np.round(1/6.0 * (order/2 + 1) * (order/2 + 2) * (2*order + 3)))
    rmn = np.zeros((num_coeffs, num_coeffs))
    r_one_d = np.zeros((num_coeffs + 1, num_coeffs + 1))

    for m in range(order+1):
        for n in range(order+1):

            if((m + n) % 2 == 0):
                coeff = np.sqrt(util.factn(n,1) * util.factn(m,1)) / np.sqrt(np.pi)

                for r in range(0,m+1,2):
                    for s in range(0,n+1,2):
                        r_one_d[m,n] += coeff * (((-1)**((s+r)/2) * 2**(n-s+m-r)) / ((util.factn(r,2)*util.factn(s,2)*util.factn(n-s,1)*util.factn(m-r,1))) * scipy.special.gamma((n-s+m-r+1) / 2.0))

    mindex = 0
    for M in range(0,order+1,2):
        for m1 in range(M+1):
            for m2 in range(M+1):
                for m3 in range(M+1):
                    if((m1+m2+m3) == M):
                        nindex = 0
                        for N in range(0,order+1,2):
                            for n1 in range(N+1):
                                for n2 in range(N+1):
                                    for n3 in range(N+1):
                                        if((n1+n2+n3) == N):
                                            rmn[mindex,nindex] = r_one_d[m1,n1]*r_one_d[m2,n2]*r_one_d[m3,n3]
                                            nindex += 1
                        mindex += 1

    return rmn


def calc_tmn(uvectors, u0, order_a, order_b):
    tmn = np.zeros((int(np.round(1/6.0 * (order_a/2 + 1) * (order_a/2 + 2) * (2*order_a + 3))), int(1 + order_a / 2)))

    tx = np.zeros((int(order_a+1), int(order_b+1)))
    ty = np.zeros((int(order_a+1), int(order_b+1)))
    tz = np.zeros((int(order_a+1), int(order_b+1)))

    psi_x = uvectors[0] / u0
    psi_y = uvectors[1] / u0
    psi_z = uvectors[2] / u0

    for m in range(int(order_a+1)):
        for n in range(0,int(2*order_b+1),2):
            if((m+n)%2 == 0):
                coeff = (np.sqrt(util.factn(m,1) * util.factn(n,1)) / np.sqrt(2 * np.pi) *
                        np.sqrt(util.factn(n,1)) / (util.factn(n,2) * u0))

                for r in range(0,m+1,2):
                    for s in range(0,n+1,2):
                        tx[m,int(n/2)] += (1 + psi_x**2) ** (-(m+n-r-s+1)/2.0) * psi_x**(n-s) * (-1)**((r+s)/2.0) * 2**((m+n-r-s)/2.0) * util.factn(m+n-r-s-1,2) / (util.factn(m-r,1) * util.factn(n-s,1) * util.factn(r,2) * util.factn(s,2))
                        ty[m,int(n/2)] += (1 + psi_y**2) ** (-(m+n-r-s+1)/2.0) * psi_y**(n-s) * (-1)**((r+s)/2.0) * 2**((m+n-r-s)/2.0) * util.factn(m+n-r-s-1,2) / (util.factn(m-r,1) * util.factn(n-s,1) * util.factn(r,2) * util.factn(s,2))
                        tz[m,int(n/2)] += (1 + psi_z**2) ** (-(m+n-r-s+1)/2.0) * psi_z**(n-s) * (-1)**((r+s)/2.0) * 2**((m+n-r-s)/2.0) * util.factn(m+n-r-s-1,2) / (util.factn(m-r,1) * util.factn(n-s,1) * util.factn(r,2) * util.factn(s,2))

                tx[m,int(n/2)] *= coeff
                ty[m,int(n/2)] *= coeff
                tz[m,int(n/2)] *= coeff


    m_index = 0
    for M in range(0,int(order_a+1),2):
        for m1 in range(M+1):
            for m2 in range(M+1):
                for m3 in range(M+1):
                    if((m1+m2+m3) == M):
                        for N in range(int(order_b+1)):
                            for n1 in range(N+1):
                                for n2 in range(N+1):
                                    for n3 in range(N+1):
                                        if((n1+n2+n3) == N):
                                            tmn[m_index, N] += tx[m1, n1] * ty[m2, n2] * tz[m3,n3]
                        m_index += 1

    return tmn


def pa_scaling(pa, param):
    pa = pa**(3*param) / (1 - 3*pa**param + 3*pa**(2*param))

    return pa

def calc_gdti_params(map_coeffs, uvecs, mask, d, D, map_order, moment_order=4):
    num_moment_coeffs = int(np.round(1/6.0 * (moment_order/2 + 1) * (moment_order/2 + 2) * (2*moment_order + 3)))
    moments = np.zeros((map_coeffs.shape[0], map_coeffs.shape[1], map_coeffs.shape[2], num_moment_coeffs))

    # Convert MAP coeffs to Moments
    ymn = calc_ymn(map_order, moment_order)

    count = 0.0
    percent_prev = 0.0
    num_vox = np.sum(mask)

    for x in range(map_coeffs.shape[0]):
        for y in range(map_coeffs.shape[1]):
            for z in range(map_coeffs.shape[2]):
                if mask[x,y,z] != 0:
                    moments[x,y,z,:] = map_coeffs_to_moments(map_coeffs[x,y,z,:], uvecs[x,y,z,:], ymn, map_order, moment_order)

                    # Update Progress
                    count += 1.0
                    percent = np.around((count / num_vox * 100), decimals = 1)
                    if(percent != percent_prev):
                        util.progress_update("Calculating GDTI Moments: ", percent)
                        percent_prev = percent

    # Convert Moments to GDTI
    tensor,_ = moments_to_HOT(moments, d, D)

    # Calculate Parameters
    mk, k_par, k_perp, fa_k = calc_dki_params(tensor, d, D)

    return mk, k_par, k_perp, fa_k

def map_coeffs_to_moments(map_coeffs, uvecs, ymn, map_order, moment_order=4):
    # Calculate scaling matrix
    u = calc_umn(uvecs, moment_order)

    # Filter MAP coefficients to reduce ringing
    map_coeffs = filter_map_coeffs(map_coeffs, map_order)

    # Multiply coeffs by ymn and scaling matrix: Moments = Map_Coeffs * Ymn * U
    moments = np.dot(np.dot(map_coeffs, ymn), u)

    return moments

def filter_map_coeffs(map_coeffs, map_order):

    index = 0
    for N in range(0,map_order+1,2):
        for n1 in range(N+1):
            for n2 in range(N+1):
                for n3 in range(N+1):
                    if((n1+n2+n3) == N):
                        #map_coeffs[index] *= (1-N/10.0)
                        #map_coeffs[index] *= (1 - 1.0 * N**2 / (N+1)**2)
                        map_coeffs[index] *= np.exp(-(N**2 / (2*4**2)))
                        index += 1

    return map_coeffs


def calc_ymn(map_order, moment_order):
    num_map_coeffs = int(np.round(1/6.0 * (map_order/2 + 1) * (map_order/2 + 2) * (2*map_order + 3)))
    moment_coeffs = int(np.round(1/6.0 * (moment_order/2 + 1) * (moment_order/2 + 2) * (2*moment_order + 3)))
    ymn = np.zeros((num_map_coeffs, moment_coeffs))

    y_one_d = np.zeros((map_order+1, moment_order+1))

    for m in range(map_order+1):
        for n in range(moment_order+1):

            coeff = np.sqrt(util.factn(m,1))
            if(((m+n) % 2) == 0):
                for r in range(0,m+1,2):
                    y_one_d[m,n] += coeff * ((-1)**(r/2.0) * 2.0**(m-r)) / (util.factn(r,2) * util.factn(m-r,1)) * scipy.special.gamma((m+n-r+1)/2.0)

    mindex = 0
    for M in range(0,map_order+1,2):
        for m1 in range(M+1):
            for m2 in range(M+1):
                for m3 in range(M+1):
                    if((m1+m2+m3) == M):
                        nindex = 0
                        for N in range(0,moment_order+1,2):
                            for n1 in range(N+1):
                                for n2 in range(N+1):
                                    for n3 in range(N+1):
                                        if((n1+n2+n3) == N):
                                            ymn[mindex,nindex] = y_one_d[m1,n1] * y_one_d[m2,n2] * y_one_d[m3,n3]
                                            nindex += 1
                        mindex += 1

    return ymn

def calc_umn(uvec, moment_order):
    moment_coeffs = int(np.round(1/6.0 * (moment_order/2 + 1) * (moment_order/2 + 2) * (2*moment_order + 3)))
    u = np.zeros((moment_coeffs, moment_coeffs))

    index = 0
    for N in range(0,moment_order+1,2):
        for n1 in range(N+1):
            for n2 in range(N+1):
                for n3 in range(N+2):
                    if((n1+n2+n3) == N):
                        u[index,index] = uvec[0]**n1 * uvec[1]**n2 * uvec[2]**n3 * np.sqrt((2.0**N)/(np.pi**3))
                        index += 1

    return u

def moments_to_HOT(moments, d, D):
    qq = np.zeros((moments.shape[0], moments.shape[1], moments.shape[2], 21))
    dd = np.zeros((moments.shape[0], moments.shape[1], moments.shape[2], 21))

    # Convert from um to mm
    rr = moments[:,:,:,1:7]
    rrrr = moments[:,:,:,7:]

    # Compute cumulants "qq" from moments
    qq[:,:,:,0] = rr[:,:,:,5]
    qq[:,:,:,1] = rr[:,:,:,4]
    qq[:,:,:,2] = rr[:,:,:,3]
    qq[:,:,:,3] = rr[:,:,:,2]
    qq[:,:,:,4] = rr[:,:,:,1]
    qq[:,:,:,5] = rr[:,:,:,0]
    qq[:,:,:,6] = rrrr[:,:,:,14] - 3*rr[:,:,:,5]**2                                         # Q_xxxx = <xxxx>-3<xx>^2
    qq[:,:,:,7] = rrrr[:,:,:,13] - 3*rr[:,:,:,5]*rr[:,:,:,4]                                # Q_xxxy = <xxxy>-3<xx><xy>
    qq[:,:,:,8] = rrrr[:,:,:,12] - 3*rr[:,:,:,5]*rr[:,:,:,3]                                # Q_xxxz = <xxxz>-3<xx><xz>
    qq[:,:,:,9] = rrrr[:,:,:,11] - rr[:,:,:,5]*rr[:,:,:,2] - 2*rr[:,:,:,4]**2               # Q_xxyy = <xxyy>-<xx><yy>-2<xy>^2
    qq[:,:,:,10] = rrrr[:,:,:,10] - rr[:,:,:,5]*rr[:,:,:,1] - 2*rr[:,:,:,4]*rr[:,:,:,3]     # Q_xxyz = <xxyz>-<xx><yz>-2<xy><xz>
    qq[:,:,:,11] = rrrr[:,:,:,9] - rr[:,:,:,5]*rr[:,:,:,0] - 2*rr[:,:,:,3]**2               # Q_xxzz = <xxzz>-<xx><zz>-2<xz>^2
    qq[:,:,:,12] = rrrr[:,:,:,8] - 3*rr[:,:,:,4]*rr[:,:,:,2]                                # Q_xyyy = <xyyy>-3<xy><yy>
    qq[:,:,:,13] = rrrr[:,:,:,7] - rr[:,:,:,2]*rr[:,:,:,3] - 2*rr[:,:,:,4]*rr[:,:,:,1]      # Q_xyyz = <xyyz>-<yy><xz>-2<xy><yz>
    qq[:,:,:,14] = rrrr[:,:,:,6] - rr[:,:,:,0]*rr[:,:,:,4] - 2*rr[:,:,:,3]*rr[:,:,:,1]      # Q_xyzz = <xyzz>-<zz><xy>-2<xz><yz>
    qq[:,:,:,15] = rrrr[:,:,:,5] - 3*rr[:,:,:,0]*rr[:,:,:,3]                                # Q_xzzz = <xzzz>-3<zz><xz>
    qq[:,:,:,16] = rrrr[:,:,:,4] - 3*rr[:,:,:,2]**2                                         # Q_yyyy = <yyyy>-3<yy>^2
    qq[:,:,:,17] = rrrr[:,:,:,3] - 3*rr[:,:,:,2]*rr[:,:,:,1]                                # Q_yyyz = <yyyz>-3<yy><yz>
    qq[:,:,:,18] = rrrr[:,:,:,2] - rr[:,:,:,2]*rr[:,:,:,0] - 2*rr[:,:,:,1]**2               # Q_yyzz = <yyzz>-<yy><zz>-2<yz>^2
    qq[:,:,:,19] = rrrr[:,:,:,1] - 3*rr[:,:,:,1]*rr[:,:,:,0]                                # Q_yzzz = <yzzz>-3<yz><zz>
    qq[:,:,:,20] = rrrr[:,:,:,0] - 3*rr[:,:,:,0]**2                                         # Q_zzzz = <zzzz>-3<zz>^2


    # Calculate Diffusion Times
    td2 = D - 1/3.0 * d
    td4 = D - 3/5.0 * d

    # Convert Cumulants to High Order Tenssors
    dd[:,:,:,0:6] = qq[:,:,:,0:6] / (2 * td2)
    dd[:,:,:,6:22] = qq[:,:,:,6:22] / (24 * td4)

    return dd, qq

def calc_dki_params(tensor, d, D):
    # Scale Kurtosis tensor
    trace = (tensor[:,:,:,0] + tensor[:,:,:,3] + tensor[:,:,:,5]) / 3.0

    scale = (4.0 * (D - 1.0/3 * d)**2 * trace**2.0)
    for i in range(6,21):
        tensor[:,:,:,i] /= scale

    # Use parameters defined by Hui et al. : Towards better MR characterization of neural tissues using directional diffusion kurtosis analysis
    k = np.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], 3))
    k[:,:,:,0] = (trace ** 2) / (tensor[:,:,:,0] ** 2) * tensor[:,:,:,6]
    k[:,:,:,1] = (trace ** 2) / (tensor[:,:,:,3] ** 2) * tensor[:,:,:,16]
    k[:,:,:,2] = (trace ** 2) / (tensor[:,:,:,5] ** 2) * tensor[:,:,:,20]
    k[np.isnan(k)] = 0

    mk = (k[:,:,:,0] + k[:,:,:,1] + k[:,:,:,2]) / 3
    k_par = k[:,:,:,2]
    k_perp = (k[:,:,:,0] + k[:,:,:,1]) / 2

    fa_k = np.sqrt(1.5 * ((k[:,:,:,0]-mk)**2 + (k[:,:,:,1]-mk)**2 + (k[:,:,:,2]-mk)**2) / (k[:,:,:,0]**2 + k[:,:,:,1]**2 + k[:,:,:,2]**2))

    return mk, k_par, k_perp, fa_k

def fit_map_glyphs(coeffs, uvectors, eigen_vectors, order, mask, moment=2):
    # Allocate for Output
    num_harmonics = int((order + 1) * (order + 2) / 2)
    SH_coeffs = np.zeros((coeffs.shape[0], coeffs.shape[1], coeffs.shape[2], num_harmonics))

    # Get Sample directions
    file_location = os.path.dirname(__file__)
    sample_dirs = np.array(util.read_direction_file(file_location + "/../direction_files_qsi/642vertices.txt"))

    # Used for Progress update
    count = 0.0
    percent_prev = 0.0
    num_vox = np.sum(mask)

    odf = np.zeros((coeffs.shape[0], coeffs.shape[1], coeffs.shape[2], sample_dirs.shape[0]))
    for x in range(odf.shape[0]):
        for y in range(odf.shape[1]):
            for z in range(odf.shape[2]):
                if mask[x,y,z] != 0:

                    om_x = sample_dirs[:,0] / uvectors[x,y,z,0]
                    om_y = sample_dirs[:,1] / uvectors[x,y,z,1]
                    om_z = sample_dirs[:,2] / uvectors[x,y,z,2]

                    mag_A = uvectors[x,y,z,0] ** 2 * uvectors[x,y,z,1] ** 2 * uvectors[x,y,z,2] ** 2

                    rho = 1.0 / np.sqrt(om_x ** 2 + om_y ** 2 + om_z **2)

                    alpha = 2 * rho * om_x
                    beta =  2 * rho * om_y
                    gamma = 2 * rho * om_z

                    scale_factor = rho ** (moment + 3) / np.sqrt(2**(2-moment) * np.pi**3 * mag_A)

                    index = 0
                    for N in range(0,order+1,2):
                        for n1 in range(0,order+1):
                            for n2 in range(0,order+1):
                                for n3 in range(0,order+1):
                                    if (n1+n2+n3) == N:
                                        C = np.zeros((sample_dirs.shape[0]))

                                        for i in range(0,n1+1,2):
                                            for j in range(0,n2+1,2):
                                                for k in range(0,n3+1,2):
                                                    C += (-1)**((i+j+k)/2.0) * scipy.special.gamma((3+moment+N-i-j-k) / 2.0) * (alpha**(n1-i) * beta**(n2-j) * gamma**(n3-k)) / (util.factn(n1-i,1) * util.factn(n2-j,1) * util.factn(n3-k,1) * util.factn(i,2) * util.factn(j,2) * util.factn(k,2))

                                        odf[x,y,z,:] += coeffs[x,y,z,index] * np.sqrt(util.factn(n1,1) * util.factn(n2,1) * util.factn(n3,1)) * C
                                        index += 1

                    odf[x,y,z,:] *= scale_factor
                    odf[x,y,z,:] /= np.amax(odf[x,y,z,:])

                    eig_vecs = eigen_vectors[x,y,z,:,:].T
                    dirs = np.matmul(eig_vecs, sample_dirs.T).T

                    # Fit ODF to SH for display
                    SH_order = util
                    SH_coeffs[x,y,z,:] = SH.fit_to_SH_MAP(odf[x,y,z,:], dirs, order)

                    # Update Progress
                    count += 1.0
                    percent = np.around((count / num_vox * 100), decimals = 1)
                    if(percent != percent_prev):
                        util.progress_update("Calculating ODFs: ", percent)
                        percent_prev = percent

    return SH_coeffs
