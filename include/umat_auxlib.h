#ifndef UMAT_AUXLIB_H
#define UMAT_AUXLIB_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Invoke a PyTorch model from Fortran UMAT
 * 
 * This function is the Abaqus-facing IPC client entrypoint.
 * Thread-safe initialization is performed on first call.
 * The shape of F is fixed as 3x3, while output sizes are model-defined.
 * In all cases F is a 3x3 deformation gradient tensor and psi is a scalar.
 * Cauchy and DDSDDE are filled from server response payload exactly as produced
 * by the TorchScript model (for example, 4 and 16 values for a 2D formulation,
 * or 6 and 36 values for a 3D formulation).
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
 *   101 - Failed to load model on server
 *   102 - Failed to insert model into server cache
 *   105 - Server inference or conversion error
 *   110 - Invalid input parameters
 *   111 - Unsupported tensor layout/shape mismatch
 *   120 - IPC connect error
 *   121 - IPC write error
 *   122 - IPC read error
 *   123 - IPC protocol error
 */
int invoke_pt(
    const char* module_filename,
    const double* F,
    const double* mat_par,
    int n_mat_par,
    double* psi,
    double* Cauchy,
    double* DDSDDE
);

/**
 * @brief Invoke a PyTorch model from Fortran VUMAT with a batch of material points.
 *
 * The deformation gradient and stress arrays are passed in Fortran layout:
 *   - defgradF(nblock, ndir+2*nshr)
 *   - stressNew(nblock, ndir+nshr)
 *
 * @return int Error code (0 = success)
 */
int invoke_pt_vumat_batch(
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

#endif /* UMAT_AUXLIB_H */
