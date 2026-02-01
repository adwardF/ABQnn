#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <map>
#include <mutex>
#include <shared_mutex>

#include <ctime>

#include <torch/torch.h>
#include <torch/script.h> 

#ifdef _WIN32
#include <windows.h>
#endif

#include <filesystem>

#include "abqnn_config.h"

int main() {
    printf("ABQnn Load Test\n");
    std::string module_filename_str("test_NH_3D.pt");

    // need to load torch_cuda.dll if using CUDA tensors
    // LoadLibraryExA("torch_cuda.dll", NULL, LOAD_WITH_ALTERED_SEARCH_PATH);

    try {
        std::filesystem::current_path(ABQNN_MODEL_PATH);
        printf("Changed working directory to: %s\n", std::filesystem::current_path().string().c_str());
        printf("Trying to load: %s\n", module_filename_str.c_str());
        torch::jit::Module module = torch::jit::load( module_filename_str );
        printf("Loaded: %s\n", module_filename_str.c_str());

        // Basic torch test
        // double; cuda device
        
        torch::Tensor F = torch::eye(3, torch::kDouble);
        F[0][0] = 1.2;
        F[0][1] = 1.05;
        
        auto results = module.forward({ F, torch::tensor({1.0, 10.0}, torch::kDouble) });
        auto result_tuple = results.toTuple();
        double psi_value =
            result_tuple->elements()[0].toTensor().item<double>();
        auto &Cauchy_result = result_tuple->elements()[1].toTensor();
        auto &DDSDDE_result = result_tuple->elements()[2].toTensor();

        printf("Results:\n");
        printf("  Strain energy (psi): %f\n", psi_value);
        printf("  Cauchy stress (Voigt): [");
        for (int i = 0; i < 6; ++i) {
            printf("%f", Cauchy_result[i].item<double>());
            if (i < 5) printf(", ");
        }
        printf("]\n");
        for( int i = 0; i < 6; ++i ) {
            printf("  DDSDDE[%d]: [", i);
            for (int j = 0; j < 6; ++j) {
                printf("%f", DDSDDE_result[i][j].item<double>());
                if (j < 5) printf(", ");
            }
            printf("]\n");
        }
    }
    catch(const std::exception& e)
    {
        printf("Exception: %s\n", e.what());
        return 1;
    }
                
    return 0;
}