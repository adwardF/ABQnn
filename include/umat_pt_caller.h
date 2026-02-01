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

#ifdef __cplusplus
}
#endif

#endif /* UMAT_PT_CALLER_H */
