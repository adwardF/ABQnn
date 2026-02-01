#ifndef UMAT_AUXLIB_H
#define UMAT_AUXLIB_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Invoke a PyTorch model from Fortran UMAT
 * 
 * This function handles dynamic library loading and delegates to pt_module_invoke.
 * Thread-safe initialization is performed on first call.
 * The shapes of the input and output arrays depdend on the material model.
 * In all cases F is a 3x3 deformation gradient tensor
 * psi is a scalar strain energy
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
 *   0 - Success
 *   1 - Failed to set DLL directory
 *   2 - Failed to load UMAT_pt_caller.dll
 *   3 - Failed to get DLL handle
 *   4 - Failed to get function address
 *   Other codes from pt_module_invoke
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

#ifdef __cplusplus
}
#endif

#endif /* UMAT_AUXLIB_H */
