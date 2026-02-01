/**
 * @file pt_caller_test.cpp
 * @brief Simple test for pt_module_invoke function
 */

#include <iostream>
#include <cstring>
#include <cmath>

#include "umat_pt_caller.h"

int main(int argc, char* argv[])
{
    std::cout << "ABQnn PT Caller Test" << std::endl;
    std::cout << "====================" << std::endl;
    
    // Default model path
    const char* model_path = "test_NH_3D.pt";
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "Testing with model: " << model_path << std::endl;
    
    // Create identity deformation gradient
    double F[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    
    // Add small deformation
    F[0][0] = 1.1;  // 10% stretch in x
    F[1][1] = 1.05; // 5% stretch in y
    F[2][2] = 1.0 / (1.1 * 1.05); // Maintain incompressibility
    
    double psi = 0.0;
    double Cauchy6[6] = {0};
    double DDSDDE[6][6] = {0};
    
    double mat_par[2] = {1.0, 10.0};
    
    int err = pt_module_invoke(model_path, &F[0][0], mat_par, 2, &psi, Cauchy6, &DDSDDE[0][0]);
    
    if (err != 0) {
        std::cerr << "Error: pt_module_invoke returned " << err << std::endl;
        return err;
    }
    
    // Print results
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Strain energy (psi): " << psi << std::endl;
    
    std::cout << "  Cauchy stress (Voigt): [";
    for (int i = 0; i < 6; ++i) {
        std::cout << Cauchy6[i];
        if (i < 5) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    for( int i = 0; i < 6; ++i ) {
        std::cout << "  DDSDDE[" << i << "]: [";
        for (int j = 0; j < 6; ++j) {
            std::cout << DDSDDE[i][j];
            if (j < 5) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\nTest completed successfully!" << std::endl;
    
    return 0;
}
