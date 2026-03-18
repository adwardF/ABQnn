#ifndef UMAT_PT_CALLER_H
#define UMAT_PT_CALLER_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #ifdef UMAT_PT_CALLER_EXPORTS
        #define UMAT_PT_CALLER_API __declspec(dllexport)
    #else
        #define UMAT_PT_CALLER_API __declspec(dllimport)
    #endif
#else
    #define UMAT_PT_CALLER_API
#endif

/**
 * @brief Invoke a PyTorch model for material computation
 * 
 * The shapes of the input and output arrays depdend on the material model.
 * In all cases F is a 3x3 deformation gradient tensor, psi is a scalar strain energy.
 * Plane stress: Cauchy: [3], DDSDDE: [3][3]
 * Plane strain: Cauchy: [4], DDSDDE: [4][4]
 * 3D stress: Cauchy: [6], DDSDDE: [6][6]
 * 
 * @param module_filename Path to the .pt TorchScript model file
 * @param F Deformation gradient tensor [3][3] (input)
 * @param psi Strain energy density (output)
 * @param Cauchy Cauchy stress in Voigt notation (output)
 * @param DDSDDE Material tangent stiffness matrix (output)
 * @return int Error code (0 = success)
 * 
 * Error codes:
 *   0   - Success
 *   101 - Failed to load module
 *   102 - Failed to insert module into cache
 *   105 - Error during model inference or result conversion
 *   106 - Unexpected type for psi result
 *   110 - Invalid input parameters (null pointers)
 */
UMAT_PT_CALLER_API int pt_module_invoke(
    const char* module_filename,
    const double* F,
    const double* mat_par,
    int n_mat_par,
    double* psi,
    double* Cauchy,
    double* DDSDDE
);

/**
 * @brief Invoke a PyTorch model for Abaqus/Explicit VUMAT-style batched computation.
 *
 * Input deformation gradients are provided in VUMAT component ordering as a
 * Fortran-style 2D array: defgradF(nblock, ndir + 2*nshr).
 *
 * Supported component layouts:
 *   - 3D (ndir=3, nshr=3): [F11, F22, F33, F12, F23, F31, F21, F32, F13]
 *   - 2D (ndir=3, nshr=1): [F11, F22, F33, F12, F21]
 *
 * The TorchScript model is expected to return a tuple:
 *   (enerInternNew, stressNew)
 * where:
 *   - enerInternNew is scalar/tensor with nblock entries,
 *   - stressNew is tensor with nblock*(ndir+nshr) values.
 *
 * @param module_filename Path to the .pt TorchScript model file
 * @param defgradF Fortran-order deformation gradient batch, shape (nblock, ndir+2*nshr)
 * @param nblock Number of material points in this call
 * @param ndir Number of direct tensor components
 * @param nshr Number of shear tensor components
 * @param mat_par Material parameters vector
 * @param n_mat_par Number of material parameters
 * @param enerInternNew Internal energy per unit mass, shape (nblock)
 * @param stressNew Stress tensor in VUMAT ordering, Fortran-order shape (nblock, ndir+nshr)
 * @return int Error code (0 = success)
 *
 * Error codes:
 *   0   - Success
 *   101 - Failed to load module
 *   102 - Failed to insert module into cache
 *   105 - Error during model inference or result conversion
 *   110 - Invalid input parameters (null pointers or invalid dimensions)
 *   111 - Unsupported tensor layout or output shape mismatch
 */
UMAT_PT_CALLER_API int pt_module_invoke_vumat_batch(
    const char* module_filename,
    const double* defgradF,
    int nblock,
    int ndir,
    int nshr,
    const double* mat_par,
    int n_mat_par,
    double* enerInternNew,
    double* stressNew
);

#ifdef __cplusplus
}
#endif

#endif /* UMAT_PT_CALLER_H */
